# XYZ inputs

[中文](README.zh.md)

Coordinates are in Angstrom and each monomer's atom order must match the AO centers of its user-supplied orbitals.
The aggregate contains contiguous fragment blocks in MEBF row order, with corresponding monomer atom order; it defines target rotations and translations.
Stage 02 applies the same transform to XYZ and RasOrb and builds the actual downstream `aggregate.xyz`.
Ethene monomer/dimer/trimer coordinates originate from the earlier ethene preparation examples.
Butadiene uses the established GS0 ethene/butadiene RICD reference; the f/f/g demonstration translates the two ethene and one butadiene carriers by 0, 3.6 and 7.2 Angstrom along z.
These illustrative aggregate placements are not claimed to be optimized equilibrium structures.
