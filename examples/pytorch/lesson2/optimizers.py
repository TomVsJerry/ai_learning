"""同一个参数、同一个梯度，SGD 和 Adam 各改一步。

分数 = 3w，损失 = (分数 - 5)²。w 从 1 出发，梯度是 -12。

运行：

    python optimizers.py
"""

import torch
from torch import nn


class OneWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0]))

    def forward(self):
        return 3 * self.w


def one_step(name, optimizer_cls):
    model = OneWeight()
    optimizer = optimizer_cls(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    score = model()
    loss = (score - 5) ** 2
    loss.backward()
    print(
        name,
        "改之前",
        model.w.item(),
        "分数",
        score.item(),
        "损失",
        loss.item(),
        "梯度",
        model.w.grad.item(),
    )
    optimizer.step()
    print("  改之后", round(model.w.item(), 4))


def main():
    one_step("SGD ", torch.optim.SGD)
    one_step("Adam", torch.optim.Adam)


if __name__ == "__main__":
    main()
