"""输出是前向算出来的数。梯度是参数加一点时，这个数的变化率。

运行：

    python output_and_grad.py
"""

import torch


def refreshment(ice_weight: float) -> float:
    sweet, ice, tea = 8.0, 2.0, 6.0
    return 0.0 * sweet + ice_weight * ice + 0.5 * tea + 1.0


def loss_from_like_bias(bias: float) -> float:
    like = 0.4 * 6.0 + 0.1 * 8.0 + bias
    dislike = -0.2 * 6.0 + 0.0 * 8.0 + 0.5
    scores = torch.tensor([[like, dislike]])
    return torch.nn.functional.cross_entropy(scores, torch.tensor([0])).item()


print("第一层输出 清爽、甜腻", [refreshment(1.0), 8.0])
print("第二层输出 喜欢、不喜欢", [0.4 * 6 + 0.1 * 8, -0.2 * 6 + 0.5])

step = 0.001
before = refreshment(1.0)
after = refreshment(1.0 + step)
print(f"冰的权重 1 加 {step}，清爽从 {before} 变成 {after}")
print("变化率", (after - before) / step)

loss_step = 0.0001
loss_before = loss_from_like_bias(0.0)
loss_after = loss_from_like_bias(loss_step)
print(f"喜欢的底分加 {loss_step}，损失从 {loss_before:.7f} 变成 {loss_after:.7f}")
print("损失的变化率", (loss_after - loss_before) / loss_step)
