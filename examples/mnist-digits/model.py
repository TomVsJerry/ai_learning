"""手写数字网络。

输入是一张图拉直后的 784 个亮度。
中间一层把 784 个数收成 128 个检测。
输出一层把 128 个检测收成 10 个分数，对应数字 0 到 9。

这里不写 Softmax。训练用的交叉熵损失会在内部把分数变成百分比。
分开写是为了数值稳定；百分比最大的那个类别，和分数最大的那个类别是同一个。
"""

import torch
from torch import nn


class DigitClassifier(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        # 笔记写成 y = x W1 + b1。按这个写法，W1 是 784×128，b1 长度是 128。
        # PyTorch 把同一份数存成 weight，形状是 (128, 784)，计算时先转置再乘。
        # 每一格亮度都按自己的权重混进每一个检测。
        self.hidden = nn.Linear(28 * 28, hidden_size)
        # 笔记里的 W2 是 128×10，b2 长度是 10。
        # PyTorch 的 weight 形状是 (10, 128)。
        # 10 个分数分别是「像 0」到「像 9」。
        self.output = nn.Linear(hidden_size, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images 是 (64, 1, 28, 28)：第 0 轴 64 是这一批有几张图，后面是 1 个通道和 28×28 个亮度。
        # view 不改亮度的数值，只把每张图排成一行。images.size(0) 保住 64，-1 收成 784。
        flat = images.view(images.size(0), -1)  # (64, 784)

        # hidden 对每一行做 y = x W1 + b1：784 个亮度收成 128 个检测。64 行共用同一份 W1、b1。
        # ReLU 把负数改成 0，形状仍是 (64, 128)。
        detected = torch.relu(self.hidden(flat))

        # output 对每一行做 y = x W2 + b2：128 个检测收成 10 个分数，下标 0 到 9。
        # 第 0 轴仍是「第几张图」，所以出来是 (64, 10)。
        scores = self.output(detected)
        return scores
