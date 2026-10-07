# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest production checkpoint: optimizer step 169.

- optimizer warmup complete: true
- supervised tokens seen: 11,075,584 / 500,000,000
- fraction of primary budget: 2.2151168%
- checkpoint SHA-256:
  `ccce01d29686a76edfe52fbaa531cf198a82bfd8aa17ded34625b8c6f14b979e`
- state SHA-256:
  `175ebccef8707fda0af0e65cf0646ac12f72b6464ab54c20ac6fa81be25124e6`
- post-warmup result SHA-256:
  `7069138a1910643eb577008076fdbafab2c6cfe52b74861539074d788df6a563`
- pretraining complete: false
- public release eligibility: not qualified

## Post-step169 validation

Byte-identical 300k validation pack:

- English: 6.210462363 -> 6.117310139 (-1.499924%)
- Thai: 6.872962925 -> 6.745732500 (-1.851173%)
- technical: 6.412851243 -> 6.337900238 (-1.168763%)
- combined: 6.498758844 -> 6.400314292 (-1.514821%)
- final holdout touched: false
- gate: PASS

Tracked hashes:

- validation step169:
  `eb6489be84dbf2ff06d7ff46905a2a6c9f84bb303c1053552c174e443881d962`
- step153-to-step169 comparison:
  `0f01fef42571bb756cad34ed00b9d3eb36ca7240788dd70443c51baa6b1b631f`

## Active work

Record and exact-qualify complete step169 evidence before any new training
authorization.

Do not execute step170 until the evidence commit is exact-qualified and
remote-synchronized. A larger bounded post-warmup tranche may be considered
only after that boundary is frozen.
