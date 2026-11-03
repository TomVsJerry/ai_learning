"""用奶茶网络把 train.py 里的 PyTorch 概念走一遍。

数字和 docs/transformer/00-neural-network-framework.md 相同，可以手算。
这里不下载 MNIST。读数字项目里的同一批类，在脚本后半段用空图对一下形状。

运行：

    python pytorch_tour.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import DigitClassifier


class TeaNet(nn.Module):
    """和 DigitClassifier 同一写法：__init__ 里登记层，forward 里算。"""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.0]]))
            self.hidden.bias.copy_(torch.tensor([1.0, 0.0]))
            self.output.weight.copy_(torch.tensor([[0.4, 0.1], [-0.2, 0.0]]))
            self.output.bias.copy_(torch.tensor([0.0, 0.5]))

    def forward(self, drinks: torch.Tensor) -> torch.Tensor:
        detected = torch.relu(self.hidden(drinks))
        return self.output(detected)


def section(title: str) -> None:
    print(f"\n== {title}")


def main() -> None:
    section("1. Tensor：带形状的一张表")
    drink = torch.tensor([[8.0, 2.0, 6.0]])
    print("小明", drink.tolist(), "shape", tuple(drink.shape), "dtype", drink.dtype, "device", drink.device)

    pixels = torch.arange(8, dtype=torch.float32).view(1, 1, 2, 4)
    print("一张 1×2×4 的小图", pixels.view(1, -1).tolist())

    section("2. Module 与 Linear：笔记的 W 存在 weight 的转置位置")
    model = TeaNet()
    for name, value in model.named_parameters():
        print(f"{name:16} shape {tuple(value.shape)}  requires_grad {value.requires_grad}")

    scores = model(drink)
    print("分数 [喜欢, 不喜欢]", [round(v, 4) for v in scores[0].tolist()])

    section("3. 损失：CrossEntropyLoss 吃分数和类别编号")
    label = torch.tensor([0])
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(scores, label)
    probabilities = torch.softmax(scores.detach(), dim=1)[0]
    print("百分比", [round(v, 6) for v in probabilities.tolist()], "加总", round(float(probabilities.sum()), 6))
    print("损失", round(loss.item(), 6), "Python 类型", type(loss.item()).__name__)

    hot = torch.tensor([[8.0, 2.0, 6.0], [0.0, -6.0, 2.0]])
    hot_labels = torch.tensor([0, 1])
    batch_loss = loss_fn(model(hot), hot_labels)
    print("两杯的平均损失", round(batch_loss.item(), 6))

    section("4. backward：每个权重旁边放上同形状的梯度")
    model.zero_grad()
    loss_fn(model(drink), label).backward()
    print("output.bias.grad", [round(v, 6) for v in model.output.bias.grad.tolist()])
    print("hidden.weight.grad")
    for row in model.hidden.weight.grad.tolist():
        print("  ", [round(v, 6) for v in row])

    section("5. zero_grad 与 step：用一个数看清改参数")
    weight = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=0.01)
    optimizer.zero_grad()
    score = 3 * weight
    sgd_loss = (score - 5) ** 2
    sgd_loss.backward()
    print("改之前 w", weight.item(), "损失", sgd_loss.item(), "梯度", weight.grad.item())
    optimizer.step()
    print("SGD 一步之后 w", round(weight.item(), 4))

    weight2 = torch.tensor([1.0], requires_grad=True)
    weight2.grad = torch.zeros_like(weight2)
    ((3 * weight2 - 5) ** 2).backward()
    ((3 * weight2 - 5) ** 2).backward()
    print("连续两次 backward、中间没有清梯度，梯度变成", weight2.grad.item())

    fresh = torch.tensor([1.0], requires_grad=True)
    adam = torch.optim.Adam([fresh], lr=0.01)
    adam.zero_grad()
    ((3 * fresh - 5) ** 2).backward()
    adam.step()
    print("同样的起点，Adam 一步之后 w", round(fresh.item(), 4))

    section("6. DataLoader：一次拿出一批，最后一批可以更短")
    drinks = torch.tensor([[8.0, 2.0, 6.0], [0.0, -6.0, 2.0], [1.0, 9.0, 2.0]])
    labels = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(drinks, labels), batch_size=2, shuffle=False)
    for batch_index, (batch_x, batch_y) in enumerate(loader, start=1):
        print(f"第 {batch_index} 批  x {tuple(batch_x.shape)}  y {batch_y.tolist()}")

    section("7. no_grad 与 eval：测量时不算梯度")
    model.eval()
    with torch.no_grad():
        held = model(drink)
    print("train 模式", model.training, "分数的 grad_fn", held.grad_fn)
    model.train()
    recorded = model(drink)
    print("train 模式", model.training, "分数的 grad_fn", type(recorded.grad_fn).__name__)

    section("8. manual_seed：同一颗种子，新建线性层的随机权重相同")
    torch.manual_seed(0)
    first = nn.Linear(3, 2)
    torch.manual_seed(0)
    second = nn.Linear(3, 2)
    print("第一格", round(first.weight[0, 0].item(), 6), round(second.weight[0, 0].item(), 6))

    section("9. 读数字网络：同一套对象，格子更多")
    torch.manual_seed(0)
    digit = DigitClassifier()
    count = sum(parameter.numel() for parameter in digit.parameters())
    blank = torch.zeros(2, 1, 28, 28)
    digit_scores = digit(blank)
    print("参数个数", count)
    print("hidden.weight", tuple(digit.hidden.weight.shape), "output.weight", tuple(digit.output.weight.shape))
    print("两张空白图的分数", tuple(digit_scores.shape))
    print("state_dict 的名字", list(digit.state_dict()))


if __name__ == "__main__":
    main()
