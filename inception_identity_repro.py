#!/usr/bin/env python3
"""Minimal, dependency-free reproducer for response-identity instability on an
OpenAI-compatible chat-completions endpoint.

Fires N byte-identical requests and reports, per call and in aggregate:

  * the `model` string the server echoed back
  * the `system_fingerprint` field, if the server sets one
  * the `x-request-id` response header and the body's `id`
  * wall-clock latency
  * whether the answer text was stable across identical requests

The question it is built to answer is narrow: **given one unchanged request,
does this endpoint describe itself consistently, and does it answer
consistently?** An evaluation harness that must record "which exact build
produced this verdict" needs yes to the first. A harness that must reproduce a
published number needs yes to the second.

Standard library only -- no `openai`, no `requests`, no venv. Python 3.8+.

Set the key without writing it to your shell history -- the inline
`export KEY=...` form lands in ~/.zsh_history and ~/.bash_history verbatim,
and this script deliberately accepts no --key argument because argv is
readable by any process via `ps`:

    zsh:  read -rs "INCEPTION_API_KEY?Inception key: " && export INCEPTION_API_KEY
    bash: read -rsp "Inception key: " INCEPTION_API_KEY && export INCEPTION_API_KEY

    python3 inception_identity_repro.py --n 40

    # any OpenAI-compatible endpoint:
    python3 inception_identity_repro.py \
        --base-url https://api.example.com/v1 --model some-model --n 40

Exit status is 0 whether or not instability is found -- this is a measurement
tool, not a test. Use --fail-on-instability if you want it to gate CI.

Written for an external evaluation of Mercury 2 (2026-08). Provider-neutral by
construction: nothing below is specific to any vendor beyond the default
--base-url and --model, both overridable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

DEFAULT_BASE_URL = "https://api.inceptionlabs.ai/v1"
DEFAULT_MODEL = "mercury-2"

# A deliberately dull, deterministic-looking prompt. The point is not to test
# reasoning -- it is to hold the request constant so that any variation in the
# response is attributable to the server. Keep it short so the run is cheap and
# so output-length variance does not dominate the latency figures.
DEFAULT_PROMPT = "Reply with exactly one word: OK"


def build_ssl_context(ca_bundle=None):
    """A verifying TLS context, with a working trust store on machines whose
    Python has none.

    A framework Python installed from python.org on macOS does not use the
    system keychain and ships with an EMPTY trust store until the bundled
    `Install Certificates.command` has been run, so every HTTPS request fails
    with CERTIFICATE_VERIFY_FAILED / "unable to get local issuer certificate".
    That is a property of the interpreter, not of the endpoint being measured.

    Resolution order: an explicit --ca-bundle, then $SSL_CERT_FILE, then
    certifi if it happens to be importable, then the system default. There is
    deliberately no flag to disable verification: this tool sends a bearer
    token, and an unverified TLS connection would put that token on the wire
    for anyone able to intercept it. Fixing the trust store is the only
    supported answer.
    """
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    env_bundle = os.environ.get("SSL_CERT_FILE")
    if env_bundle and os.path.exists(env_bundle):
        return ssl.create_default_context(cafile=env_bundle)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


TLS_HELP = """
The TLS handshake failed before any request reached the endpoint, so nothing
was measured and nothing was billed. This is a local trust-store problem, not
a finding about the API.

  macOS, Python from python.org -- run the installer's certificate script once:
      /Applications/Python\\ 3.13/Install\\ Certificates.command
      (substitute your version; `python3 -V` prints it)

  or point this script at any CA bundle you already have:
      pip install certifi          # then re-run; it is picked up automatically
      python3 -m pip show certifi  # to find the path
      ... --ca-bundle /path/to/cacert.pem
      or: export SSL_CERT_FILE=/path/to/cacert.pem

Verification is never disabled by this script: the request carries a bearer
token, and an unverified connection would expose it.
"""


def post(url, payload, headers, timeout, ctx):
    """One POST. Returns (status, response_headers, parsed_body_or_None, err)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            body = r.read().decode("utf-8", "replace")
            # dict(r.headers) preserves the header names the server actually
            # sent; matching is done case-insensitively at the call site.
            return r.status, dict(r.headers), json.loads(body), None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return e.code, dict(e.headers or {}), None, body[:400]
    except Exception as e:  # network, TLS, timeout, malformed JSON
        return None, {}, None, f"{type(e).__name__}: {e}"


def runs_test(seq):
    """Wald-Wolfowitz runs test on a two-valued sequence.

    Counting transitions alone cannot distinguish "the two labels alternate
    because each request is routed independently" from "the labels happen to
    clump". This does: under independent per-call labelling the number of runs
    has a known mean and variance, so a z near zero means the observed pattern
    is what independent per-request assignment looks like, and a large negative
    z means the labels are clustered in time (which a redeploy would produce).

    Returns (runs, expected_runs, sd, z), or None if the sequence is not
    two-valued or is too short for the normal approximation.
    """
    vals = sorted(set(seq))
    if len(vals) != 2:
        return None
    n1 = seq.count(vals[0])
    n2 = seq.count(vals[1])
    n = n1 + n2
    if n1 < 5 or n2 < 5 or n < 20:   # normal approximation not trustworthy
        return None
    runs = 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    mu = 2 * n1 * n2 / n + 1
    var = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))
    if var <= 0:
        return None
    sd = var ** 0.5
    return runs, mu, sd, (runs - mu) / sd


def header(hdrs, name):
    """Case-insensitive header lookup; HTTP header names are not case-sensitive
    and intermediaries do not agree on casing."""
    for k, v in hdrs.items():
        if k.lower() == name.lower():
            return v
    return None


def main():
    p = argparse.ArgumentParser(
        description="Measure response-identity stability across N identical requests.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=40, help="number of identical requests")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=None,
                   help="omitted entirely if unset, so the server default applies")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds between requests (0 = as fast as the server allows)")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--key-env", default="INCEPTION_API_KEY",
                   help="environment variable holding the API key")
    p.add_argument("--ca-bundle", default=None,
                   help="PEM trust store to verify the endpoint against "
                        "(default: $SSL_CERT_FILE, then certifi, then the system store)")
    p.add_argument("--abort-after", type=int, default=3, metavar="K",
                   help="stop if the first K calls all fail before any succeeds "
                        "(0 disables); prevents a misconfigured client from "
                        "grinding through the whole run")
    p.add_argument("--out", default=None, help="write the full per-call record to this JSON file")
    p.add_argument("--fail-on-instability", action="store_true",
                   help="exit 1 if more than one distinct model string is observed")
    args = p.parse_args()

    key = os.environ.get(args.key_env)
    if not key:
        # Never accept a key as a command-line argument: argv is visible to any
        # process on the machine via `ps`, and in an interactive shell it lands
        # in the shell history file.
        # The two shells spell the silent-prompt form differently, and getting it
        # wrong fails quietly -- zsh parses the prompt as a SUFFIX on the variable
        # name ("NAME?prompt"), so the bash spelling ("-p prompt" then NAME) reads
        # into the wrong variable and leaves $NAME empty, which lands right back
        # here. Both spellings are printed rather than guessed at.
        sys.exit(f"error: ${args.key_env} is not set.\n"
                 f"  set it without writing it to your shell history:\n"
                 f"    zsh:  read -rs \"{args.key_env}?API key: \" && export {args.key_env}\n"
                 f"    bash: read -rsp \"API key: \" {args.key_env} && export {args.key_env}\n"
                 f"  then check it took, without printing it:\n"
                 f"    echo \"${{#{args.key_env}}} chars\"")

    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}

    print(f"endpoint    {url}")
    print(f"model       {args.model!r}   requests: {args.n}")
    print(f"prompt      {args.prompt!r}")
    print(f"temperature {'server default (omitted)' if args.temperature is None else args.temperature}")
    print()

    ctx = build_ssl_context(args.ca_bundle)

    calls = []
    aborted = None
    t_start = time.time()
    for i in range(args.n):
        if args.sleep and i:
            time.sleep(args.sleep)
        t0 = time.monotonic()
        status, hdrs, body, err = post(url, payload, headers, args.timeout, ctx)
        latency = time.monotonic() - t0
        rec = {
            "index": i,
            "t_offset_s": round(time.time() - t_start, 3),
            "latency_s": round(latency, 3),
            "status": status,
            "error": err,
            "request_id": header(hdrs, "x-request-id"),
            "response_id": (body or {}).get("id"),
            "model": (body or {}).get("model"),
            "system_fingerprint": (body or {}).get("system_fingerprint"),
        }
        # `is not None`, not a truth test: an empty JSON object is falsy, and
        # treating it as "no body" left `text` unset on a 200 response, which
        # then raised KeyError in the summary and threw away the whole run's
        # data at the final step. A degenerate response is a measurement too.
        if body is not None:
            try:
                msg = body["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                msg = ""
            rec["finish_reason"] = (body["choices"][0] or {}).get("finish_reason") \
                if body.get("choices") else None
            rec["text"] = msg
            rec["text_sha256"] = hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16]
            rec["usage"] = body.get("usage")
        calls.append(rec)
        flag = "" if status == 200 else f"  <-- HTTP {status} {err or ''}"
        print(f"  [{i + 1:>3}/{args.n}] {latency:6.3f}s  model={rec['model']!r:<34} "
              f"req_id={rec['request_id'] or '-'}{flag}")

        # Fail fast on a client-side misconfiguration. A bad trust store, a
        # wrong base URL, or a rejected key fails identically on every call, so
        # continuing costs the operator time and (on an auth failure) sends the
        # key another N-K times for no measurement. Only trips while NOTHING has
        # succeeded yet -- a failure after a good response is real data about the
        # endpoint and must not stop the run.
        if args.abort_after and len(calls) >= args.abort_after \
                and not any(c["status"] == 200 for c in calls):
            aborted = (f"aborted after {len(calls)} consecutive failures with no "
                       f"successful response")
            print(f"\n{aborted} -- the remaining "
                  f"{args.n - len(calls)} calls were not attempted.")
            break
    duration = time.time() - t_start

    ok = [c for c in calls if c["status"] == 200]
    print(f"\n{'=' * 78}\n{len(ok)}/{len(calls)} attempted requests succeeded "
          f"in {duration:.1f}s\n")
    if not ok:
        print("No successful responses -- nothing was measured.")
        errs = " ".join(str(c["error"] or "") for c in calls)
        if "CERTIFICATE_VERIFY_FAILED" in errs or "SSLCertVerificationError" in errs:
            print(TLS_HELP)
        elif any(c["status"] in (401, 403) for c in calls):
            print(f"  The endpoint rejected the credential. Check that "
                  f"${args.key_env} holds the whole key:\n"
                  f"      echo \"${{#{args.key_env}}} chars\"")
        else:
            print("  Check the key, the base URL, and network reachability.")
        return 0

    # --- the headline: does the server describe itself consistently? ---------
    models = [c["model"] for c in ok]
    counts = Counter(models)
    print(f"distinct `model` strings returned: {len(counts)}")
    for m, n in counts.most_common():
        print(f"    {n:>4}x  {m!r}")

    # A stable endpoint that was redeployed mid-run would show ONE transition,
    # at the redeploy. Many transitions mean the labels are interleaved across
    # concurrently-serving backends, which no timestamp-based split can undo.
    transitions = sum(1 for a, b in zip(models, models[1:]) if a != b)
    if len(counts) > 1:
        print(f"\n    transitions between labels, in call order: {transitions}")
        print(f"    (a single mid-run redeploy would give exactly 1; "
              f"{transitions} indicates interleaving)")
        for m in counts:
            idxs = [c["index"] for c in ok if c["model"] == m]
            print(f"    {m!r} spans call {min(idxs)}..{max(idxs)}")
        rt = runs_test(models)
        if rt:
            runs, mu, sd, z = rt
            print(f"\n    Wald-Wolfowitz runs test: {runs} runs, "
                  f"{mu:.1f} expected (sd {sd:.1f}) if each call were labelled "
                  f"independently\n    z = {z:+.2f} -- "
                  + ("consistent with per-request assignment across "
                     "concurrently-serving backends"
                     if abs(z) < 2 else
                     "NOT consistent with independent per-request assignment; "
                     "the labels are clustered in time"))

    fps = Counter(c["system_fingerprint"] for c in ok)
    print(f"\n`system_fingerprint` values: "
          + ", ".join(f"{v!r} x{n}" for v, n in fps.most_common()))
    if set(fps) == {None}:
        print("    -> not implemented by this endpoint: there is no build identifier")
        print("       to pin an evaluation to, independent of the `model` string above.")

    rids = [c["request_id"] for c in ok if c["request_id"]]
    print(f"\n`x-request-id` present on {len(rids)}/{len(ok)} successful responses"
          + ("" if rids else "  -> absent: calls are not traceable in provider logs"))
    if rids:
        print(f"    first: {rids[0]}\n    last:  {rids[-1]}")

    # --- and does it answer consistently? -----------------------------------
    #
    # Guard first. On a reasoning model, the reasoning tokens are drawn from the
    # SAME max_tokens budget as the visible answer, and they are spent first. Set
    # max_tokens too low and every call returns HTTP 200 with empty content and
    # finish_reason="length" -- at which point "distinct answers: 1" is not
    # stability, it is the same truncation 60 times, and reporting it as
    # stability would be a false negative on the exact question this section
    # asks. Detected and named rather than left for the reader to infer.
    empty = [c for c in ok if not (c.get("text") or "").strip()]
    truncated = [c for c in ok if c.get("finish_reason") == "length"]
    no_completion = [c for c in ok
                     if isinstance(c.get("usage"), dict)
                     and c["usage"].get("completion_tokens") == 0]
    if len(empty) == len(ok) and (truncated or no_completion):
        reas = [c["usage"].get("reasoning_tokens") for c in ok
                if isinstance(c.get("usage"), dict)
                and isinstance(c["usage"].get("reasoning_tokens"), int)]
        print(f"\n[!] answer stability NOT MEASURED: all {len(ok)} responses were "
              f"empty.\n"
              f"    {len(truncated)}/{len(ok)} finished with reason 'length'"
              + (f"; reasoning consumed a median of {statistics.median(reas):.0f} "
                 f"of the {args.max_tokens}-token budget" if reas else "")
              + f".\n    Re-run with a larger budget before reading anything into "
                f"answer consistency:\n"
                f"        --max-tokens {max(256, args.max_tokens * 16)}\n"
                f"    (That every call truncates identically is itself worth "
                f"noting: the\n     endpoint returns 200 with no content and no "
                f"signal beyond finish_reason.)")
    else:
        texts = Counter((c.get("text") or "").strip() for c in ok)
        print(f"\ndistinct answers to the identical prompt: {len(texts)}")
        for t, n in texts.most_common(6):
            show = t if len(t) <= 60 else t[:57] + "..."
            print(f"    {n:>4}x  {show!r}")
        if len(texts) > 1:
            modal = texts.most_common(1)[0][1]
            print(f"    -> modal answer covers {modal}/{len(ok)} calls; the remainder "
                  f"differ under an unchanged request")
        if empty:
            print(f"    [!] {len(empty)}/{len(ok)} of those were empty "
                  f"({len(truncated)} truncated at the token budget)")

    # --- is the billed input count stable under an unchanged request? --------
    # Not part of the original design; added after a run showed it moving. An
    # identical request should bill an identical number of input tokens, so any
    # spread here is a property of the server's accounting, not of the prompt.
    pt = [c["usage"].get("prompt_tokens") for c in ok
          if isinstance(c.get("usage"), dict)
          and isinstance(c["usage"].get("prompt_tokens"), int)]
    if pt:
        ptc = Counter(pt)
        print(f"\nbilled input tokens for the identical prompt: "
              + ", ".join(f"{v} x{n}" for v, n in sorted(ptc.items())))
        if len(ptc) > 1:
            print(f"    -> {len(ptc)} distinct counts (min {min(pt)}, max {max(pt)}) "
                  f"for one unchanged request;\n       the billed input size is not "
                  f"a function of the input alone.")

    lat = sorted(c["latency_s"] for c in ok)
    print(f"\nlatency s: min {lat[0]:.3f}  median {statistics.median(lat):.3f}  "
          f"p95 {lat[min(len(lat) - 1, int(0.95 * len(lat)))]:.3f}  max {lat[-1]:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "meta": {
                    "base_url": args.base_url, "model": args.model, "n": args.n,
                    "prompt": args.prompt, "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "started_unix": t_start, "duration_s": duration,
                    "script_version": "1.2",
                    "attempted": len(calls),
                    "aborted": aborted,
                },
                "summary": {
                    "succeeded": len(ok),
                    "distinct_model_strings": len(counts),
                    "model_string_counts": dict(counts),
                    "label_transitions": transitions,
                    "distinct_answers": len(texts),
                    "system_fingerprint_values": {str(k): v for k, v in fps.items()},
                    "request_id_present": len(rids),
                },
                "calls": calls,
            }, f, indent=2)
        print(f"\nfull per-call record written to {args.out}")

    if args.fail_on_instability and len(counts) > 1:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
