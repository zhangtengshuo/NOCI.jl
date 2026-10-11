# PySCF 原生输入模板

`cas.py.in` 是原生 DF-CASSCF 求解器输入，SCF 和 CASSCF 都显式使用 `cas.auxiliary_basis`，读取用户提供的原生 MO `.npy`，按声明的电子数、自旋和根求解并导出原始 `.npz` 数据。
`integrals_dfcd.py.in` 是原生 AO DF/CDERI 输入，显式使用 `integrals.auxiliary_basis` 和度量阈值，分块输出三中心因子 HDF5 及一体 AO 数据，不生成四指标 ERI。
这些文件提供可编辑的真实求解器输入，本版的标准流程仍拒绝启动 PySCF 后端，不能把单独的模板当作已完成 NOCI 适配器。

PySCF CAS 辅助基的配置来自 [DF-CASSCF 接口](https://pyscf.org/_modules/pyscf/mcscf/df.html#density_fit)，分块 DF 使用 [cholesky_eri_b 接口](https://pyscf.org/_modules/pyscf/df/outcore.html#cholesky_eri_b) 的 ED 度量处理和 lindep 参数。
缺省 CAS 与积分辅助基相同；只显式填写一处时另一处继承，用户也可分别显式指定不同辅助基。
PySCF 的 CASPT2 适配器尚未实现，不能将其他 PT2 方法当作 CASPT2 自动执行。

| 占位符 | 含义 |
| --- | --- |
| `XYZ_LITERAL`、`BASIS_LITERAL`、`AUXILIARY_BASIS_LITERAL`、`INITIAL_ORBITALS_LITERAL` | 后续适配器应提供安全引用的 Python 字符串字面量。 |
| `CHARGE`、`SPIN_2S`、`NCAS`、`NELECAS_ALPHA`、`NELECAS_BETA`、`NCORE`、`ROOT` | 用户给定的逐态 PySCF CAS 参数。 |
| `THREADS`、`MEMORY_MB` | 后续适配器确定的线程数和 PySCF 内存设置。 |
| `METRIC_LINDEP` | 用户的辅助库仑度量小特征值舍弃条件。 |
| `STATE_OUTPUT_LITERAL`、`CDERI_PATH_LITERAL`、`ONE_AO_PATH_LITERAL`、`OVERLAP_PATH_LITERAL` | 后续适配器确定的阶段原始产物路径。 |

原始 `.npz` 需要后续适配器写成具有 AO、轨道、CI 和来源信息的标准状态 bundle；原始 AO 因子 HDF5 需要在当前 project 的实际公共轨道基上变换，处理冻结核心并导出 NOCI 因子来源元数据。
这些转换、资源预检和数值验收仍属于预留后端的实施工作，不由本版模板替代。
