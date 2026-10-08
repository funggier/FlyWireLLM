# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest production checkpoint candidate: optimizer step 233.

- supervised tokens seen: 15,269,888 / 500,000,000
- fraction of primary budget: 3.0539776%
- checkpoint SHA-256:
  `1513602d920fe57b7bb12ee7ee69cdf77ea23f39f0253cdabb718e55413dd964`
- state SHA-256:
  `93f5256d1fa82d29dcfb95c997518afd46901101475af4c553816e42c2edf7ed`
- post-warmup-3 result SHA-256:
  `e26e181ec16be56a23155544fa5f907e64bc5fd85ea47a39abacc45ef691b12e`
- optimizer warmup complete: true
- pretraining complete: false
- public release eligibility: not qualified

## Post-step233 validation

Byte-identical 300k validation pack:

- English: 5.931827876 -> 5.791585690 (-2.364232%)
- Thai: 6.452203094 -> 6.260309313 (-2.974082%)
- technical: 6.140651583 -> 6.005837822 (-2.195431%)
- combined: 6.174894184 -> 6.019244275 (-2.520689%)
- final holdout touched: false
- gate: PASS

Tracked hashes:

- validation step233:
  `89ab8cfc6bc0ed2f3923f53407ba9d432d5a76ac0086497a17545d79d6913561`
- step201-to-step233 comparison:
  `cd9c4f78e74200de9af3731b9cbeb4f3fecb07b5b9c2118c64348837bba687de`

## Active work

Record and exact-qualify complete step233 evidence before any new training
authorization.

Do not execute step234 until the evidence commit is exact-qualified and
remote-synchronized. The next post-warmup tranche size will be chosen from
the two consecutive successful 32-update validation gates at step201 and
step233.
