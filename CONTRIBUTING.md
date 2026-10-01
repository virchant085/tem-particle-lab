# 测试与贡献说明

请先阅读 [README](README.md) 的零基础教程。详细参数与命令见[进阶说明](docs/user_guide_advanced.md)。问题反馈使用 GitHub 的“测试反馈”模板，包含 commit SHA、系统/GPU、复现步骤、预期结果和实际错误。

不要提交或上传以下内容：未经授权的 TEM 原始视频、人工标注、正式模型权重、数据集快照、个人运行日志、虚拟环境，以及含本机用户名或实验路径的文件。项目的 `.gitignore` 已排除常见目录，但提交前仍应运行 `git status` 检查。

提交代码前运行：

```powershell
.\.venv-ml\Scripts\python.exe -X utf8 -m pytest -q
.\.venv-ml\Scripts\python.exe -X utf8 -m compileall -q src scripts
```

修改前端后还要运行：

```powershell
node --check ui\learning.js
```

请将新功能、错误修复和实验参数调整放在独立分支，通过 pull request 合并。方法或阈值变化需要说明使用的数据划分、评价指标和失败案例，不能用观测率代替准确率。
