# 02: Position fragments and rotate orbitals

[中文](README.zh.md)

Require the completed stage-01 RasOrb/VecDet bindings and stage-00 aggregate poses.
Fit corresponding atoms using a proper Kabsch rotation, preserve the carrier's internal geometry, and write per-instance XYZ, rotation matrices and the aggregate geometry.
Run each instance's C1 reference calculation to obtain RunFile/OneInt, then rotate all requested state orbitals with that same rotation matrix.
Outputs are `<project>/aggregate.xyz`, per-instance reference data and instance/state `.orb` files.
All downstream CommonOrb and integral tasks use this aggregate geometry; stage 02 neither optimizes structures nor changes the user's active space.

```bash
bash run.sh ethene_dimer.yaml all
```

`prepare` performs the small geometry fit, `compute` launches the reference tasks, and `finalize` performs numerical orbital rotation.
