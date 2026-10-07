# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint candidate: optimizer step 104.

- supervised tokens seen: 6,815,744 / 500,000,000
- fraction of primary budget: 1.3631488%
- checkpoint SHA-256:
  `0f61a87d497543af516f3818d2d47ce21412fec009ac7a08492ee4396952726b`
- external state SHA-256:
  `e2a98f2466436c163cc519082ee7e718c1fe55ec1621296337392a5d720cdd89`
- eighth-tranche result SHA-256:
  `caecd23b8b79aba1f6e2e76e1d81152387ff3186be0fe5ba37d0a268dd2ca6e6`
- pretraining complete: false
- public release eligibility: not qualified

## Rolling checkpoint retention

- retained eighth-tranche full checkpoints: 1
- historical eighth-tranche checkpoints pruned: 15
- step89-104 metrics retained: 16 / 16
- final checkpoint reload: PASS

## Post-step104 validation gate

Byte-identical 300k validation pack:

- English: 6.991381387 -> 6.776078678 (-3.079545%)
- Thai: 7.424897029 -> 7.331277076 (-1.260892%)
- technical: 7.127175571 -> 6.907709166 (-3.079290%)
- combined: 7.181151329 -> 7.005021640 (-2.452666%)
- final holdout touched: false
- gate: PASS

Tracked hashes:

- validation step104:
  `4652344185a436f3185626859773b78efe49ddf517c373538c87694df935aa60`
- step88-to-step104 comparison:
  `7f03b35b296f2ee23863ad52ccf3f5b480c1be5fed725886a3dd61421e22a109`

## Active work

Record and exact-qualify the complete step104 evidence before any new
training authorization.

Do not execute step105 until the evidence commit is exact-qualified and
remote-synchronized. Keep the next tranche warmup-aware because the declared
optimizer warmup continues through step153.
