# 安装内默认输入模板

这里提供标准准备入口在用户流程没有模板时使用的默认文件。
用户可编辑的标准副本和占位符说明位于 [examples/ahpc_noci_standard/00_system_define/templates](../../../examples/ahpc_noci_standard/00_system_define/templates/README.zh.md)，用户文件优先，默认文件不覆盖用户副本。

OpenMolcas 的 CAS、可选 CASPT2、片段／聚集体 C1 参考及 RICD／CD 积分均通过模板渲染，不在 schema-4 驱动中另维护硬编码输入正文。
PySCF 原生 CAS／AO DF 模板作为后续适配器输入契约，当前能力仍标记 reserved，不在 CPU NOCI 安装中强制导入 Python 或 PySCF。
