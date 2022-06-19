# simple-lstm-server

## 安装依赖

*推荐使用 Conda 环境。*

```shell
# 新建一个 Conda 环境
conda create -n lstm python=3.6
# 激活这个环境
conda activate lstm
# ======= 如果不使用 Conda 环境，直接在 Python 3.6 环境下执行下面 ========
# 安装依赖
pip install -r requirements.txt
```

## 运行

### 运行服务器

运行需要的参数可以在 `server.py` 中修改。

```shell
python server.py
```

服务器功能由 NodeJS 后端调用。

### 运行测试

```shell
python test.py
```