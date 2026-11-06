"""SimpleNN(nn.Module) 的创建、打印和一次前向。

输入两个数。fc1 收成两个数，ReLU 之后 fc2 收成一个数。

运行：

    python simple_nn.py
"""

import torch
from torch import nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def fill(model: SimpleNN) -> None:
    with torch.no_grad():
        model.fc1.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        model.fc1.bias.copy_(torch.tensor([1.0, 0.0]))
        model.fc2.weight.copy_(torch.tensor([[0.4, 0.1]]))
        model.fc2.bias.copy_(torch.tensor([0.0]))


def main() -> None:
    model = SimpleNN()
    fill(model)
    print(model)
    features = torch.tensor([[8.0, 2.0], [0.0, -6.0]])
    # 构造已在 SimpleNN() 完成。这里走 Module.__call__，张量传入 forward。
    by_call = model(features)
    by_forward = model.forward(features)
    print("输入形状", tuple(features.shape))
    print("输出形状", tuple(by_call.shape))
    # (2, 1) 收成一根轴，再变成 Python 列表 [2.0, 0.0]。
    print("model(x)", by_call.view(-1).tolist())
    print("model.forward(x) 相同", torch.equal(by_call, by_forward))
    for name, value in model.named_parameters():
        print(name, tuple(value.shape))


if __name__ == "__main__":
    main()
