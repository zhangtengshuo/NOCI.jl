# 05：AHPC-NOCI

运行前：确认 03/04 产物，生成并解析检查 `<project>.yaml`，固定使用 `ahpc-noci` 和 `integrals.backend: factorized`。
运行：以明确的 Julia/BLAS 线程和工作区设置调用 NOCI 生产接口，默认使用 CPU。
需要 GPU 时，先用 `--withgpu` 安装，再在阶段 00 设置 `runtime.use_gpu: true`。
运行后：确认最终 JSON 和实时进度的完成事件，保留 `.out`、矩阵快照和可选断点。

进入 NOCI 的片段态标签只有 `f_99_S1` 等短名称，体系长名称只出现在算例标题和输出根名。
用户在 00 修改态表或筛选设置后，既有完成结果不能被直接跳过复用，应采用新 `project` 或明确清理受影响的算例阶段。

```bash
bash run.sh ethene_dimer.yaml prepare
bash run.sh ethene_dimer.yaml compute
bash run.sh ethene_dimer.yaml finalize
```
