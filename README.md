# mercury2 — response-identity and reproducibility probe

A dependency-free reproducer, and the raw records from 240 calls, for three
behaviours of the Mercury 2 chat-completions API that affect clients which
need to reproduce, cache, or attribute a response.

Everything here is a measurement of the **API surface** as served on
2026-08-04. Nothing in this repository is a claim about the model's quality,
about diffusion models, or about any vendor other than the endpoint named in
the default arguments — which are overridable, because the script is
provider-neutral and works against any OpenAI-compatible endpoint.

## Why this exists

I was adding Mercury 2 as a judge to an evaluation harness. Harnesses of that
kind make three assumptions that most APIs satisfy quietly: a response can be
cached and re-fetched under the same key, a result can be attributed to a
specific served version, and an identical request produces an identical
response. Each of those turned out to need checking here, so I checked them
properly rather than reporting an impression.

The cost result is worth stating first, because it is the reason any of this
matters: on my workload Mercury 2 was roughly **twelve times cheaper per call
than the most expensive model I compared it against, at sub-second median
latency.** That is a real advantage. The three items below are the things
standing between it and a class of customer that cannot currently adopt it.

## What was run

Four arms of 60 calls, each arm holding one request completely fixed.

| | arm A | arm B | arm C | arm D |
|---|---|---|---|---|
| prompt | `Reply with exactly one word: OK` | same | same | `List the first five prime numbers…` |
| `max_tokens` | 16 | 16 | 400 | 400 |
| pacing | burst | 1 s apart | burst | burst |
| HTTP 200 | 60/60 | 60/60 | 60/60 | 60/60 |
| `mercury-2` label share | 16/60 | 14/60 | 7/60 | 9/60 |
| label transitions | 19 | 22 | 14 | 16 |
| runs-test *z* | −1.49 | +0.20 | +1.06 | +0.36 |
| `system_fingerprint` | no value ×60 | no value ×60 | no value ×60 | no value ×60 |
| `x-request-id` | 60/60 unique | 60/60 unique | 60/60 unique | 60/60 unique |
| distinct answers | not measured¹ | not measured¹ | 1 | 2 |
| median latency | 0.388 s | 0.417 s | 0.472 s | 0.603 s |

¹ Arms A and B ran at `max_tokens=16`. On this model the reasoning tokens are
drawn from the same budget as the visible answer and are spent first, so all
120 calls returned HTTP 200 with `finish_reason: "length"`,
`completion_tokens: 0`, and empty content — with no error signal. Those two
arms remain valid for the label question, which does not depend on the
response body, and are void for answer stability. The script now detects this
condition and prints `answer stability NOT MEASURED` instead of a misleading
count. The failure is left in the record rather than re-run away.

## Finding 1 — the returned `model` string is not stable, and it varies per request

Across 240 calls the API returned `inception/mercury-2-prod-h100` on 194 and
`mercury-2` on 46.

The two ordinary explanations are both ruled out by the data:

A **mid-run redeployment** would produce exactly one boundary between the two
labels. Each 60-call arm produced 14 to 22 transitions.

**Burst routing** predicts fewer transitions once calls are spaced out. Arm B
spaced them one second apart and produced *more* transitions than the
back-to-back arm A (22 against 19).

To avoid eyeballing that, each arm gets a Wald–Wolfowitz runs test. Under
independent per-call labelling the number of runs has a known mean and
variance, so this distinguishes "assigned independently per request" from
"clustered in time". All four arms sit within about 1.5 standard deviations of
the independent-assignment mean (z = −1.49, +0.20, +1.06, +0.36). Nothing here
looks like an epoch boundary.

**The consequence for a client:** because the label is assigned per request, no
timestamp split of a run's records can attribute an output to a backend, and a
re-run cannot be constrained to the same one.

## Finding 2 — there is no `system_fingerprint` field in the response

No value came back on any of the 240 calls. The probe originally read the field
with a defaulting lookup, which returns the same thing whether the key is
present and null or missing from the body altogether, so the records in `data/`
cannot separate those two cases. Version 1.3 records the distinction, and a
call under 1.3 resolves it: **the key is absent from the response body**. That
is consistent with a published response schema which does not list the field.

Two limits on that resolution, since finding 1 constrains what a single call
can establish. It is one call, and it returned under the
`inception/mercury-2-prod-h100` label — so I have not observed a body under the
`mercury-2` label at 1.3 and am not claiming the two return the same shape.

The consequence holds regardless: this is the standard OpenAI-compatible field
a client reaches for to pin a served configuration, there is nothing there, and
with finding 1 no mechanism remains for identifying what produced a result.

## Finding 3 — an identical request does not always return identical bytes

Arm C returned the byte-identical string `OK` on all 60 calls. Arm D, on a
different prompt, returned two distinct strings under an unchanged request:

```
2,3,5,7,11        51/60
2, 3, 5, 7, 11     9/60
```

Both are correct. The difference is whitespace, which is precisely why it is
worth reporting: substantively identical answers that differ in bytes break
exact-match scoring, output hashing, and any cache keyed on response text.
Because `temperature` is documented with a floor of 0.5 and no `seed` is
exposed, there is no client-side setting that suppresses it.

The label counts (51/9) and the answer counts (51/9) match, which invites the
reading that one backend emits one format. **It does not hold.** The `h100`
label produced 42 compact and 9 spaced responses; the `mercury-2` label
produced 9 compact and none spaced; 18 of 60 calls violate the mapping. The
spaced form appeared only under the `h100` label, but most `h100` calls
returned the compact form, so the label does not determine the answer.

## A smaller observation, offered as-is

Billed `completion_tokens` varied (10, 11, 12) across arm D calls whose output
text was byte-identical, and arms A–C billed three different `prompt_tokens`
values for one unchanged prompt. **No claim is made that this affects
invoices** — the relationship between these counters and what is charged is
not visible from outside and was not investigated, and at one or two tokens
the cost is negligible. It is recorded only because a token count cannot be
used as a fingerprint of an unchanged request.

One tidier description of this did not survive testing. Arms A–C looked like a
single per-call offset applied to every counter at once; arm D was run
specifically to test that on a prompt it was not derived from, and
`prompt_tokens` was constant at 15 across all 60 calls. The offset description
is confined to that one prompt and is not a property of the endpoint. What
replicated out of sample is only the narrower output-side statement above.

## What already works

`x-request-id` was present and unique on all 240 responses. Every call in
`data/` can therefore be located in provider-side logs, which is what makes
these records useful to anyone able to look at the other end.

It is the one field here that already does its job, and it is not mentioned in
the public documentation — so a client has no basis for relying on it and no
way to know it is there. Documenting it would cost a paragraph and is the
cheapest item on this page.

## Three suggestions, ranked by implementation cost

1. **Stabilize the `model` string, or document what it means.** By far the
   cheapest. If the two strings denote the same weights, returning one of them
   is a routing-layer change; if they do not, that is information clients need.
   Even a documentation line stating that the field is a routing label rather
   than a version identifier would be a large improvement over clients
   inferring it from record dumps.

2. **Add a populated `system_fingerprint`.** OpenAI-compatible clients reach
   for this field by default and it is not there. A value that changes when the
   served configuration changes gives clients a pinnable identity without
   committing the provider to version stability, which is the harder promise.

3. **Offer a deterministic option** — `temperature=0`, a `seed`, or an
   equivalent. The most engineering work and the largest unlock: without one
   of them, no client can make output reproducible, which rules the model out
   of every role where responses are cached, hashed, or replayed. Worth noting
   that the documented handling of an out-of-range `temperature` is a reset to
   0.75 with a `warning` in the response — so a client that passes 0 hoping for
   determinism is not refused, it is moved to the middle of the sampling range.
   I did not test that; all four arms omitted `temperature` and took the server
   default.

## Reproducing this

Single file, standard library only. No `openai`, no `requests`, no
virtualenv. Python 3.8+. A full arm is 60 calls, runs in well under a minute,
and costs a few cents.

Set the key without writing it into your shell history. The script
deliberately accepts **no** `--key` argument, because argv is readable by any
process via `ps`:

```sh
# zsh
read -rs "INCEPTION_API_KEY?API key: " && export INCEPTION_API_KEY
# bash
read -rsp "API key: " INCEPTION_API_KEY && export INCEPTION_API_KEY
```

Then:

```sh
# the four arms as run, in order
python3 inception_identity_repro.py --n 60 --max-tokens 16  --sleep 0 --out armA.json
python3 inception_identity_repro.py --n 60 --max-tokens 16  --sleep 1 --out armB.json
python3 inception_identity_repro.py --n 60 --max-tokens 400 --sleep 0 --out armC.json
python3 inception_identity_repro.py --n 60 --max-tokens 400 --sleep 0 --out armD.json \
    --prompt "List the first five prime numbers, separated by commas, and nothing else."

# any other OpenAI-compatible endpoint
python3 inception_identity_repro.py --base-url https://api.example.com/v1 --model some-model
```

Exit status is 0 whether or not instability is found — this is a measurement
tool, not a test. Use `--fail-on-instability` if you want it to gate CI. TLS
verification cannot be disabled; the request carries a bearer token, so an
unverified connection would put it on the wire. If your Python has an empty
trust store (common with a python.org framework build on macOS), the script
detects the resulting handshake failure and prints the fix rather than
reporting it as a finding about the endpoint.

## What's in `data/`

One JSON file per arm, containing run metadata and a full per-call record:
status, returned `model`, `system_fingerprint`, `x-request-id`, response `id`,
`finish_reason`, response text and its SHA-256, the full `usage` object, and
wall-clock latency. No credentials, headers, or local paths are recorded.

## Scope and limits

Four arms of 60 calls, from one client, on one network path, on one day,
against one endpoint. Latency figures are an order of magnitude, not a
benchmark. The label share moved between arms (16, 14, 7, 9 out of 60) and
three or four runs cannot establish whether that share is stable — no claim is
made about it. Everything here is exploratory and none of it is a
preregistered confirmatory result.

This came out of building [trust-eval](https://github.com/shedu5/trust-eval),
an evaluation harness for whether an LLM judge can distinguish real evidence
from fabricated evidence. That project's conclusions are separate from this
repository and are not restated here.

## License

MIT. Use the script against anything.
