## flash-attention-V2


### 文件结构

``` bash
.
├── .data                # 数据集（Multi30k）
├── bench.py             # 对齐测试
├── conf.py              # 设置模型超参数
├── data.py              # 训练数据加载
├── flash-backward.cu    # flash attention 后向
├── flash.cu             # flash attention 前向   
├── graph.py             # 绘制 LOSS 图表（训练完成后）
├── image                # 图像
├── main.cpp             # CUDA 链接用
├── models               # Pytorch 模型定义文件
├── mul_test.py          # 乘法测试（用于测试 C++ 矩阵乘法和 Pytorch 乘法结果是否一致）
├── requirements-in.txt  # Python 依赖库（输入）
├── requirements.txt     # Python 依赖库（由 pip-compile 自动生成）
├── test.py              # 前向测试
├── train.py             # 训练
└── util                 # 常用函数
```


### 运行

配置 Python 环境

```bash
conda create -n mv python=3.10
source activate mv
```

```
pip install -r requirements.txt
pip install -r requirements-in.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

训练：
```bash
python train.py
```

单步前向测试：
```bash
python test.py
```
