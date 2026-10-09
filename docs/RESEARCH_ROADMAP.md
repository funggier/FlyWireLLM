# FlyWireLLM Research Roadmap

## Project story

FlyWireLLM is intended to be understandable from the beginning, not only useful
at the end.

The project starts with a blank random-initialized language model, establishes
a conventional Transformer as a trustworthy control, records how language
capability emerges through qualified checkpoints, and only then introduces
architectural hypotheses inspired by FlyWire/connectome research.

The goal is not to attach biological terminology to a Transformer. The goal is
to turn observations from biological circuitry into falsifiable engineering
hypotheses and test whether those hypotheses improve a language model under
controlled conditions.

## Standard research budget: approximately 50M parameters

Approximately 50M parameters is the default FlyWireLLM research scale.

The current control, Base-50M-v1, contains exactly 50,213,376 parameters. This
size is deliberately retained because it is:

- large enough to exhibit meaningful language-learning behavior;
- small enough to train and repeatedly evaluate on the available local CPU
  machine;
- practical for ablations and multiple architecture variants;
- small enough that failed experiments remain affordable;
- suitable for matched-parameter and matched-compute research.

Parameter count should not be increased merely to improve headline capability.

A move above the 50M research budget should only be considered after a
FlyWire-inspired architecture demonstrates a repeatable advantage over the
control and scaling is needed to test whether that advantage survives at a
larger size.

## Frozen control

Base-50M-v1 is the conventional control:

- decoder-only causal Transformer;
- RMSNorm;
- RoPE;
- grouped-query attention;
- SwiGLU;
- tied embeddings;
- 32K SentencePiece tokenizer;
- 1,024-token training context;
- random initialization with no imported pretrained weights.

This baseline must remain independently understandable and must not be silently
modified to incorporate FlyWire-inspired mechanisms mid-training.

## FlyWire-inspired experimental families

Candidate research families are separate models built around the same
approximately-50M budget.

### Sparse connectivity

Test whether learned or structured sparse communication can preserve quality
while reducing active computation.

### Conditional routing

Test whether familiar/easy inputs can use a short path while uncertain or
unusual inputs activate deeper or additional computation.

This is related to the working intuition that skilled behavior can be mostly
automatic until an unfamiliar signal requires more deliberate processing.

### Recurrence and feedback

Test whether controlled recurrent or feedback paths improve iterative
integration compared with a strictly feed-forward stack of Transformer blocks.

### Specialized circuits

Test architectures that allow sub-networks to specialize while retaining a
clear integration path. Specialization should emerge from training rather than
being assigned biological labels without evidence.

### Local-before-global processing

Test whether local processing followed by selective/global integration can
match or improve the control with less computation than applying global
interaction uniformly.

### Gating and inhibitory-style computation

Test learnable gates that suppress unnecessary paths or signals. Biological
terminology is motivational only; artificial units must be described by their
actual mathematical operation.

## Experimental method

Each FlyWire-inspired experiment should:

1. preserve Base-50M-v1 as the control;
2. state the FlyWire/connectome observation that motivates the hypothesis;
3. change as few independent variables as practical;
4. remain near the same parameter budget;
5. report training tokens and compute explicitly;
6. use the same validation protocol where applicable;
7. keep the final holdout sealed until a genuine final decision point;
8. retain exact Git/checkpoint/data lineage;
9. include negative results;
10. avoid biological capability claims that the experiment did not measure.

A variant does not need lower loss to be useful. A matched-quality model that
uses materially less compute can also be a successful result.

## Evidence dimensions

FlyWireLLM should retain several forms of evidence rather than optimizing one
number:

- validation loss and perplexity;
- Thai, English, and technical/scientific/code balance;
- training stability and gradient behavior;
- parameter efficiency;
- active/total compute for conditional architectures;
- throughput and memory;
- deterministic checkpoint lineage;
- qualitative text-generation progression;
- later, task and reasoning evaluations appropriate to model maturity.

## Capability progression record

A fixed qualitative-generation probe should be used at selected checkpoints.
The prompt set and decoding settings must be versioned so outputs can be
compared over time.

The initial probe should cover at least:

- Thai continuation and basic question-like prompts;
- English continuation and basic question-like prompts;
- technical prose;
- code completion;
- structured formats such as JSON/Markdown;
- short context consistency.

The probe is observational evidence, not a replacement for validation and not
a license to touch the final holdout.

## Current Base-50M evidence

At optimizer step 425, cumulative supervised training was 27,852,800 tokens.

At optimizer step 681, cumulative supervised training was 44,630,016 tokens,
8.9260032% of the 500M primary budget.

At optimizer step 1193, cumulative supervised training is 78,184,448 tokens,
15.6368896% of the 500M primary budget.

Step681 to step1193 validation on the same sealed 300k validation pack:

| Category | step681 | step1193 | Relative change |
| --- | ---: | ---: | ---: |
| general English | 4.867667660 | 4.391436153 | -9.783567% |
| general Thai | 4.740044962 | 4.085647923 | -13.805714% |
| technical/scientific/code | 4.982780476 | 4.405080693 | -11.593924% |
| combined | 4.863497699 | 4.294054923 | -11.708503% |

All categories improved again. The same validation packs were used and the
final holdout remained untouched.

The second fixed qualitative-generation probe compares step681 and step1193
using the exact same 12 synthetic prompts and greedy decoding settings. Thai
surface-form behavior improves visibly: step1193 produces complete Thai clauses
and sentence-like continuations on prompts where step681 could collapse into
punctuation/list patterns. However factual correctness, semantic relevance and
repetition control remain weak. English remains sentence-shaped but can be
irrelevant or invent pseudo-words, while Python and JSON structure are still
unreliable.

The v2 probe is deliberately synthetic and observational. It reads no corpus
partition and does not touch final holdout. Its tracked artifact is
`results/l004/qualitative-probe-step681-step1193-v2.json` with SHA-256
`1262ad30117b268c22fc2d55cd1998fa3571cb1d02ab26c516951753356b7248`.

This is strong evidence that the current 50M Transformer control is still
learning materially at step1193. It is not evidence that pretraining is
complete or that the model is ready for public release or assistant use.

## Long-term sequence

The intended research sequence is:

~~~text
blank model
  -> qualified Base-50M Transformer control
  -> measured capability emergence
  -> completed/stable Base-50M control
  -> FlyWire-Sparse-50M
  -> FlyWire-Routing-50M
  -> FlyWire-Recurrent-50M
  -> FlyWire-Circuit-50M
  -> matched-resource comparisons and ablations
  -> scale only the variants that demonstrate repeatable advantages
~~~

The project should preserve why each branch was created, what hypothesis it
tested, what evidence supported or rejected it, and what was learned even when
the experiment did not win.
