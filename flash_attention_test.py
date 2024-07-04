import math
import time

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import torch.nn as nn
import flash_attention_forward

class FlashAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(FlashAttention,self).__init__()
        self.n_head=n_head
        self.attention =flash_attention_forward.forward_compute

    def forward(self, q, k, v):
        out =self.attention(q, k, v)
        return out






d_model = 512
n_head = 8
model = FlashAttention(d_model, n_head)

# 创建一个长序列
sequence_length = 32
batch_size = 1
d_model = 32
sequence = torch.randn(batch_size, n_head, sequence_length, d_model).cuda()
# 准备查询、键和值
q = sequence
k = sequence
v = sequence


start_time = time.time()
for i in range(2):
    out = model.forward(q, k, v)
torch.cuda.synchronize() if torch.cuda.is_available() else None
flash_time = time.time() - start_time

    # 打印性能差异
print(f"flash attention: {flash_time} seconds")
