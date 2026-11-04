"""同一组数经过 ReLU、Sigmoid、Tanh。形状不变，每个格子单独换算。

运行：

    python activations.py
"""

import torch
import torch.nn.functional as F

values = torch.tensor([-5.0, 0.0, 2.0])
print("线性层交出来", values.tolist())
print("F.relu     ", F.relu(values).tolist())
print("torch.relu ", torch.relu(values).tolist())
print("sigmoid    ", [round(v, 6) for v in torch.sigmoid(values).tolist()])
print("tanh       ", [round(v, 6) for v in torch.tanh(values).tolist()])

hidden = torch.tensor([-5.0, 0.0])
weight = torch.tensor([0.4, 0.1])
print("下一层直接用 [-5, 0]", float((hidden * weight).sum()))
print("下一层用 ReLU 之后 ", float((torch.relu(hidden) * weight).sum()))
