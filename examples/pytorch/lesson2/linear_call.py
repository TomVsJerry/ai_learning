"""nn.Linear(3, 2) 对小明和热浓茶各做一次加权和加底分。

运行：

    python linear_call.py
"""

import torch
from torch import nn

layer = nn.Linear(3, 2)
with torch.no_grad():
    # 第 0 行是清爽的系数，第 1 行是甜腻的系数。
    layer.weight.copy_(torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.0]]))
    layer.bias.copy_(torch.tensor([1.0, 0.0]))

drinks = torch.tensor([
    [8.0, 2.0, 6.0],
    [0.0, -6.0, 2.0],
    [0.0, 0.0, 0.0],
])
detected = layer(drinks)
by_hand = drinks @ layer.weight.t() + layer.bias

print("weight", tuple(layer.weight.shape), layer.weight.tolist())
print("bias", layer.bias.tolist())
print("小明   ", detected[0].tolist())
print("热浓茶 ", detected[1].tolist())
print("空杯   ", detected[2].tolist())
print("与 x @ weight.T + bias 相同", torch.equal(detected, by_hand))
