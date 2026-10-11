# XYZ 输入

XYZ 的坐标单位是 Angstrom，单体的原子顺序必须与对应用户轨道的 AO 中心顺序一致。
聚集体 XYZ 按 `mebf_table` 行顺序分块，每块保持对应单体的原子顺序，用于确定旋转和平移位置。
02 会把单体 XYZ 与 RasOrb 按同一位置变换，实际后续计算采用其重新拼合的 `aggregate.xyz`。

乙烯单体、二聚体和三聚体坐标来自原 `examples/ahpc_noci/00_system_define/geometry`。
丁二烯单体来自本地已有 `reference_runs/ethene_butadiene_ricd` 的 GS0 示例；f/f/g 示范位置由两个乙烯和一个丁二烯沿 z 轴分别平移 0、3.6、7.2 Angstrom 得到，不作为优化后的物理平衡结构声明。
