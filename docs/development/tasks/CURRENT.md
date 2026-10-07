# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 88.

- supervised tokens seen: 5,767,168 / 500,000,000
- checkpoint SHA-256: e5c6427334044b300c98c964499a14cb9126db87a2996b9ca5505f417e6bd27a
- step88 evidence commit: 61f969000a330d686ebbabb0d6387c6e4a283fa9
- post-step88 validation gate: PASS
- final holdout touched: false

## Eighth warmup-aware tranche

Runner commit: 79e3bd76a5088e6d9b60e3ffa647674edb78dbe6
Authorization: configs/pretraining-tranche-l004-v8.json
Authorization SHA-256: ffc8bfccec3bd0ec655d2899b3ac20417fd496fd39d2de906fc4dc18c5333367

Bound: step88 -> step104, 16 updates, 1,048,576 additional supervised tokens, rolling checkpoint retention 1, validation after step104 on the unchanged 300k validation pack. Final holdout remains untouched.

The smaller tranche is deliberate because optimizer warmup continues through step153.

Step89 is permitted only after the v8 authorization commit is exact-qualified and remote-synchronized.
