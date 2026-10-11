# OpenMolcas 输入模板

`cas.in`、`caspt2.in`、`fragment_reference.in`、`aggregate_reference.in`、`integrals_ricd.in`、`integrals_cd.in` 都是实际用于生成 OpenMolcas 输入的模板，可以在当前流程中修改。
C1 参考任务用 `group = nosym`，所有默认模板没有 EXPBAS。

| 占位符 | 来源 |
| --- | --- |
| `XYZ`、`BASIS`、`POINT_GROUP` | 当前任务的结构路径、`cas.basis` 和该分子的点群；C1 渲染成 nosym。 |
| `GATEWAY_INTEGRAL_OPTIONS`、`SEWARD_INTEGRAL_OPTIONS` | 单体 CAS 的 RICD／CD 设置，分别用于互斥的积分近似输入。 |
| `GENERATION_THRESHOLD` | 04 的 RICD 原子辅助基或分子 CD 生成阈值。 |
| `CHARGE`、`NACTEL`、`INACTIVE`、`RAS2`、`STATE_SYMMETRY`、`MULTIPLICITY`、`CIROOT`、`RLXROOT` | 用户定义的单体态参数。 |
| `INITIAL_ORBITALS`、`STATE`、`MOLECULE`、`CARRIER_GEOMETRY` | CAS 的实际种子、态名、分子种类及载体结构标识，与摆放编号无关。 |
| `FRAGMENT` | 02 C1 参考输入的片段名；共享的 01 CAS 不使用实例标识。 |
| `CASPT2_BLOCK` | 已渲染的 `caspt2.in`；禁用时为空。 |
| `IPEA_SHIFT`、`IMAGINARY_SHIFT`、`MAX_ITERATIONS` | 00 的 CASPT2 参数，shift 单位 Eh。 |

默认 CASPT2 使用 `Multistate = 1 {{RLXROOT}}`，明确计算用户选择的单个 CASSCF 根，见 [OpenMolcas CASPT2 手册](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html)。
原始 RasOrb／VecDet 在 CASPT2 前导出，后续 NOCI 使用原始 CAS 波函数与分别记录的 CASSCF／CASPT2 能量，不用 PT2ORB 替换种子或原始 CAS 轨道。
修改 RASSCF 收敛设置等原生选项可直接编辑模板，改模板会改变共享 01 的任务定义。
