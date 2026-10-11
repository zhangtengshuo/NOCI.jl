# 03: Construct the common orbital basis

[中文](README.zh.md)

Require the positioned stage-02 orbitals and selected-root monomer energies.
Prepare the aggregate C1 reference and CommonOrb control files using each project's own requested state set.
Compute the reference, then build the orbital union and transform the wavefunctions during `finalize`.
Outputs include `<project>/COMMONORB`, short instance/state `.vec/.det` names and actual frozen/deleted counts.
The internal file map carries any project prefix required by CommonOrb; final wavefunction names do not retain that long prefix.
Frozen settings come from stage 00, while deleted counts must be present in actual CommonOrb output; missing counts are never guessed as zero.

```bash
bash run.sh ethene_dimer.yaml all
```

Changing a project's state set requires its own CommonOrb and integral products, even when stage 01 can be reused.
