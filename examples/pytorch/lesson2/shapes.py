"""打印这段小网络每一层的形状，并用种子 0 走 50 步 SGD。

n_in、n_h、n_out 决定权重有多大。
batch_size 只决定这一张表有几行，不改变权重的形状。

运行：

    python shapes.py
"""

import torch
from torch import nn

torch.manual_seed(0)

n_in, n_h, n_out, batch_size = 10, 5, 1, 10

x = torch.randn(batch_size, n_in)
y = torch.tensor(
    [[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]]
)

model = nn.Sequential(
    nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out),
    nn.Sigmoid(),
)

print(f"x 形状 {tuple(x.shape)}    含义：{batch_size} 个样本，每个 {n_in} 个特征")
print(f"y 形状 {tuple(y.shape)}     含义：{batch_size} 个样本，每个 {n_out} 个目标")

hidden = model[0](x)
activated = model[1](hidden)
score = model[2](activated)
prediction = model[3](score)
print(f"Linear({n_in}, {n_h}) 之后 {tuple(hidden.shape)}")
print(f"ReLU 之后           {tuple(activated.shape)}")
print(f"Linear({n_h}, {n_out}) 之后 {tuple(score.shape)}")
print(f"Sigmoid 之后        {tuple(prediction.shape)}")

for name, param in model.named_parameters():
    print(f"参数 {name:8} 形状 {tuple(param.shape)}")

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
first_loss = None
for epoch in range(50):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if epoch == 0:
        first_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"第 0 步损失 {first_loss:.6f}")
print(f"第 49 步损失 {loss.item():.6f}")
print("另一批只有 4 个样本时，输出形状", tuple(model(torch.randn(4, n_in)).shape))
