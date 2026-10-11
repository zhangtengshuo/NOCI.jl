# 04：因子积分

运行前：读取实际公共轨道和 03 的冻结／删除计数，生成 C1 SEWARD CHH5 输入。
运行：RICD 用 GATEWAY RICD/CDThreshold，CD 用 SEWARD Cholesky/ThrCholesky；二者不混写。
运行后：调用 NOCI 的共享 HDF5 导入接口，生成 `integrals.one`、`integrals.sys`、`factors.jld2`。

没有 `&EXPBAS`，不构造常规四指标 `.two`，因子文件是 canonical-pair 行 × auxiliary 列的版本 2 bundle。
RICD 需要 兼容配置中明确接受的 OpenMolcas4NOCI v26.06.2，因子的物理来源与阈值语义由 CHH5 元数据保留。

```bash
bash run.sh ethene_dimer.yaml all
```

每个 project 在自己的实际公共轨道基上独立生成积分，不复制其他 project 的因子。
DFCD 的原生 PySCF 路线本版预留，详见根目录后端契约。
