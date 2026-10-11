# User-supplied initial orbitals

[中文](README.zh.md)

The RasOrb files are teaching seeds referenced directly by the workflow; it does not generate, choose, reorder or interpret active orbitals.
The ethene seed originates from the earlier ethene preparation example; the butadiene seed originates from the established ethene/butadiene RICD reference.
They accompany the ethene CAS(2,2) and butadiene CAS(4,4) settings here.
Provide matching seeds when changing the molecule, geometry, basis or AO order.
The reserved PySCF `user_ethene_*.npy` files are not supplied; users must provide native PySCF MO coefficients and matching AO metadata.
Renaming a RasOrb file cannot convert it into that native representation.
