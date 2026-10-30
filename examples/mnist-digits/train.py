"""训练一个读手写数字的网络，并在测试集上看它读得怎样。

流水线：

    28×28 亮度
        → 拉成 784 个数
        → 线性层 + ReLU（128 个检测）
        → 线性层（10 个分数）
        → 交叉熵损失（内部把分数变成百分比，再看正确那一类）
        → 按损失改 W1、b1、W2、b2

运行：

    python train.py
    python train.py --epochs 3
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitClassifier


DATA_DIR = Path(__file__).resolve().parent / "data"


def make_loaders(batch_size: int):
    """准备训练图和测试图。

    ToTensor 把 0～255 的亮度除以 255，收成 0～1。
    标签是 0～9 的整数，不是 10 格里放一个 1。
    交叉熵损失接受的就是这种整数：它表示正确百分比落在第几格。
    """
    to_tensor = transforms.ToTensor()
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=to_tensor)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=to_tensor)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(model, loader, loss_fn, optimizer):
    """用现有参数走完一整遍训练图，并按损失改参数。"""
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in loader:
        # 清掉上一批留在参数上的梯度，避免加到这一批上。
        optimizer.zero_grad()

        # 前向：图片 → 10 个分数。形状是 (批量, 10)。
        scores = model(images)
        # 损失 = 这一批里「正确那一类的百分比」取对数后再取负，然后求平均。
        # 百分比越接近 1，损失越接近 0。
        loss = loss_fn(scores, labels)

        # 反向：算出每个 W、b 该往哪边改，才能让损失变小。
        loss.backward()
        # 真正改参数。Adam 按各自的梯度调整步子大小。
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predicted = scores.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """在测试图上只前向、不改参数。测试图训练时没见过。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in loader:
        scores = model(images)
        loss = loss_fn(scores, labels)
        total_loss += loss.item() * labels.size(0)
        predicted = scores.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def show_one_prediction(model, loader):
    """打印测试集第一张图的 10 个百分比，方便对上 Softmax。"""
    model.eval()
    images, labels = next(iter(loader))
    scores = model(images[:1])
    # dim=1 表示在 10 个数字之间做百分比，一张图的 10 个数加起来是 1。
    probabilities = torch.softmax(scores, dim=1)[0]
    predicted = int(probabilities.argmax())
    truth = int(labels[0])

    print(f"第一张测试图的真实数字是 {truth}，读成 {predicted}")
    for digit, probability in enumerate(probabilities.tolist()):
        print(f"  {digit}: {probability:.4f}")


def main():
    parser = argparse.ArgumentParser(description="训练手写数字网络")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    train_loader, test_loader = make_loaders(args.batch_size)
    model = DigitClassifier(args.hidden_size)

    # 交叉熵：先把 10 个分数变成百分比，再取正确那一格的负对数。
    loss_fn = nn.CrossEntropyLoss()
    # 要改的就是模型里全部 W 和 b。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        print(
            f"第 {epoch} 轮  "
            f"训练损失 {train_loss:.4f}  训练读对 {train_acc:.3%}  "
            f"测试损失 {test_loss:.4f}  测试读对 {test_acc:.3%}"
        )

    show_one_prediction(model, test_loader)


if __name__ == "__main__":
    main()
