# Useful Code Repository

这是一个分享有用Python代码的库。

## 目录结构

```
src/          # 源代码目录
examples/     # 示例代码
docs/         # 文档
tests/        # 测试代码
```

## 使用说明

请将有用的Python代码按照功能分类放入相应的目录中。

## 安装依赖

使用conda创建环境（推荐）：

```bash
conda env create -f environment.yml
conda activate useful-code-repo
```

或者使用pip安装：

```bash
pip install pytest>=6.0.0
```

## 运行示例

运行随机结构生成示例：

```bash
python examples/example_usage.py
```

这将生成一个包含20个原子的测试结构，其中Ti原子占70%，Al原子占30%。生成的结构将保存为`test_structure.data`文件，格式为LAMMPS数据文件。

## 运行测试

```bash
python -m pytest tests/
```

或者

```bash
python tests/test_example.py
```