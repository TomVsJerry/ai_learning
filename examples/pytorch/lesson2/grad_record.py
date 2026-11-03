"""清爽这一格：块外留下梯度记录，no_grad 块里不留。

运行：

    python grad_record.py
"""

import torch
from torch import nn

layer = nn.Linear(3, 2)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.0]]))
    layer.bias.copy_(torch.tensor([1.0, 0.0]))

drink = torch.tensor([[8.0, 2.0, 6.0]])
print("weight.requires_grad", layer.weight.requires_grad)
print("输入 drink.requires_grad", drink.requires_grad)

recorded = layer(drink)
print("块外结果的记录", type(recorded.grad_fn).__name__)
recorded[0, 0].backward()
print("清爽对底分的梯度", layer.bias.grad.tolist())
print("清爽对第 0 行权重的梯度", layer.weight.grad[0].tolist())

with torch.no_grad():
    quiet = layer(drink)
print("no_grad 里结果的记录", quiet.grad_fn)
print("weight.requires_grad 仍是", layer.weight.requires_grad)
