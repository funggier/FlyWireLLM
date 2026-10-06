# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Qualified implementation chain

Foundation:
`c216937868d0b8391fe9ed91c4ceb8269de7e93b`

Runner:
`6b5c2946ea7b8b452e6d317de7ccd438bd23a091`

Bounded authorization:
`674354ea572532f3d8d5c0be856a43078b2e10ae`

All three were exact-qualified and remote-synchronized before production step 1.

## Production progress

First bounded tranche: DONE.

- optimizer steps: 4 / 7,630 planned;
- supervised tokens: 262,144 / 500,000,000;
- fraction of primary budget: 0.0524288%;
- final checkpoint:
  `external://FlyWireLLM-data/L004/Runs/base50m-first-tranche-v1/checkpoints/step-000004.pt`;
- checkpoint SHA-256:
  `d4ab8703cf723600e1fa6606d28674b06638284dd1f50ea2b771d6a96f947266`;
- final state SHA-256:
  `8be4f6cbe5c6b5ff84095b44aea25d418a44f449c739a5d642b8fcfc377c1c78`;
- tracked result:
  `results/l004/first-bounded-tranche-v1.json`;
- tracked result SHA-256:
  `fd906c4a7170dc3fcce6a673dc920d352559a9a9760ab31219163c18995d3908`;
- pretraining started: true;
- pretraining complete: false;
- public release eligibility: not qualified.

All four checkpoint/metric hashes were re-verified. Step 5 beyond the bounded
authorization was rejected and did not mutate state.

## Active phase

Phase J — deterministic validation gate.

Do not authorize a larger training tranche from training loss alone.

Next:

1. build deterministic validation-only Thai/English/technical evaluation data;
2. keep final holdout untouched;
3. evaluate identical validation data on deterministic step 0 and step 4;
4. compare category + combined validation loss;
5. only then decide the next bounded tranche.
