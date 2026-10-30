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
        # images 进来是 (批量, 1, 28, 28)。
        # 线性层吃的是一行特征，所以拉成 (批量, 784)。
        flat = images.view(images.size(0), -1)

        # 第一层：y = x W1 + b1。ReLU 把负数改成 0 再送出去。
        detected = torch.relu(self.hidden(flat))

        # 第二层：10 个原始分数。数值越大，越像对应的数字。
        scores = self.output(detected)
        return scores
