# 用户提供的初始轨道

这些 RasOrb 是教学种子，流程直接引用，不生成、挑选、重排或分析其中的活性轨道。
乙烯文件复制自原 `examples/ahpc_noci/00_system_define/orbitals/ethene_ccpvdz_seed.RasOrb`；丁二烯文件复制自已有 `reference_runs/ethene_butadiene_ricd/00_system_define/orbitals/butadiene_cas44_seed.RasOrb`。
它们分别配套本示例的乙烯 CAS(2,2) 和丁二烯 CAS(4,4) 设置；用户换分子、结构、基组或 AO 顺序时应自行提供匹配种子。

PySCF 预留模板中的 `user_ethene_*.npy` 没有随例提供，应由用户提供 PySCF AO 约定下的 MO 系数及配套元数据，不能通过把 RasOrb 改名获得。
