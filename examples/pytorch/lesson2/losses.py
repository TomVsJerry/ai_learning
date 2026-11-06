"""三种损失：各用一组固定的输出和答案，收成一个数。

运行：

    python losses.py
"""

import torch
from torch import nn

mse = nn.MSELoss()
predictions = torch.tensor([[0.8], [0.2]])
mse_targets = torch.tensor([[1.0], [0.0]])
print("MSE", mse(predictions, mse_targets).item())

cross_entropy = nn.CrossEntropyLoss()
scores = torch.tensor([[3.2, -0.7]])
print("CrossEntropy", cross_entropy(scores, torch.tensor([0])).item())

bce = nn.BCEWithLogitsLoss(reduction="none")
logits = torch.tensor([3.2, -0.7])
binary_targets = torch.tensor([1.0, 0.0])
each = bce(logits, binary_targets)
print("BCEWithLogits 两条", [round(v, 6) for v in each.tolist()])
print("BCEWithLogits 平均", each.mean().item())
print("内部 Sigmoid", [round(v, 6) for v in torch.sigmoid(logits).tolist()])

print("BCE(3.2, 1) 与 CE([3.2, 0], 类别 0)")
print(
    nn.BCEWithLogitsLoss()(torch.tensor([3.2]), torch.tensor([1.0])).item(),
    nn.CrossEntropyLoss()(torch.tensor([[3.2, 0.0]]), torch.tensor([0])).item(),
)
