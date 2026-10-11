# 01：共享单体 CAS 计算

运行前：检查 00 定义和种子，按分子种类／结构／态生成唯一任务，选用用户模板渲染 OpenMolcas CAS 和可选 CASPT2 输入。
运行：每个所需单体态只算一次，同分子同结构的不同摆放编号共享原始 RasOrb 和 VecDet；启用时在原始波函数导出后计算同一根的 CASPT2 能量。
运行后：确认正常结束、所选根能量及产物存在，写完成记录，并在 `bindings/<project>/` 发布该算例的实例／态引用。

任务路径如 `f/GS0/S1/`，f_99 和 f_03 都引用此任务，跨 YAML 也可复用。
换了坐标、种子、CAS 参数或基组后，既有同名任务会报冲突，不能默默拿旧结果继续。

```bash
bash run.sh ethene_dimer.yaml prepare
bash run.sh ethene_dimer.yaml compute
bash run.sh ethene_dimer.yaml finalize
```

`prepare` 不启动计算，`compute` 启动真实 OpenMolcas，`finalize` 整理结果。

01 是本流程唯一跨 project 共享的计算阶段，没有供体 project 或先后依赖，任何一个 YAML 都可以先运行。
