# 用户输入模板

这里的文件属于用户流程，可直接编辑；`prepare` 按实际选中的模板生成输入，不覆盖用户模板。
安装内 `scripts/preparation/templates/` 提供缺省文件，这里保存的是标准流程可编辑副本。

选择顺序是：YAML `templates.<role>` 明确指定的文件 → 当前 00 下 `templates/<backend>/<默认文件名>` → 安装内同名默认文件。
显式指定的路径不存在时直接报错，不偷偷换用默认；相对路径从 YAML 所在的 00 目录解析。
删掉 YAML 的 `templates` 区块仍会使用这里存在的同名文件，只有流程里没有对应文件才回退安装默认。

```yaml
templates:
  cas: templates/openmolcas/cas.in
  caspt2: templates/openmolcas/caspt2.in
  fragment_reference: templates/openmolcas/fragment_reference.in
  aggregate_reference: templates/openmolcas/aggregate_reference.in
  integrals: templates/openmolcas/integrals_ricd.in
```

| 角色 | 准备阶段 | 用途 |
| --- | --- | --- |
| `cas` | 01 | 单体 CAS 主输入，包含原始波函数导出和可选 CASPT2 插入位置。 |
| `caspt2` | 01 | 默认渲染后插入 CAS 主输入；`caspt2.enabled: false` 时省略。 |
| `fragment_reference` | 02 | 每个摆放位置的 C1 RunFile／OneInt。 |
| `aggregate_reference` | 03 | 当前 project 聚集体的 C1 RunFile／OneInt。 |
| `integrals` | 04 | 当前后端和 RICD／CD／DFCD 方法的积分输入。 |

占位符使用 `{{UPPER_CASE_KEY}}`，只做文本替换，没有 shell 展开或代码求值；OpenMolcas `$Project`、`$CurrDir` 保留给 OpenMolcas 执行。
可以增加后端支持的原生输入行；维持 C1、必要的数据导出及后续阶段的数据契约仍由用户负责。
实际使用的模板路径、大小、修改时间和小模板文本进入任务定义，修改模板后不能继续使用旧的同名任务记录。
模板没有 SHA 校验，也不触发活性空间分析。
各 project 不引用其他 project 的模板或结果。

[openmolcas/](openmolcas/README.zh.md) 说明当前可执行模板，[pyscf/](pyscf/README.zh.md) 说明预留的原生 PySCF 求解器输入与输出边界。
