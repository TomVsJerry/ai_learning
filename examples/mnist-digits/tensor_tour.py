"""把张量从一排数讲到一批小图。数字和奶茶笔记相同。

运行：

    python tensor_tour.py
"""

import torch
from torch import nn


def show(title: str, value: torch.Tensor) -> None:
    print(f"{title}")
    print(f"  数值 {value.tolist()}")
    print(f"  形状 {tuple(value.shape)}   轴的个数 {value.dim()}   格子数 {value.numel()}")


def main() -> None:
    print("== 一个数：0 个轴")
    sweet = torch.tensor(8.0)
    show("甜度", sweet)

    print("\n== 一排数：1 个轴。位置 0 甜，1 冰，2 茶")
    person = torch.tensor([8.0, 2.0, 6.0])
    show("小明", person)
    print(f"  下标 0 的甜 = {person[0].item()}")
    print(f"  下标 1 的冰 = {person[1].item()}")
    print(f"  下标 2 的茶 = {person[2].item()}")

    print("\n== 一张表：2 个轴。轴 0 选哪一杯，轴 1 选哪一项原料")
    table = torch.tensor([
        [8.0, 2.0, 6.0],
        [0.0, -6.0, 2.0],
    ])
    show("两杯", table)
    print(f"  table[0, 1] 小明的冰 = {table[0, 1].item()}")
    print(f"  table[1, 0] 热浓茶的甜 = {table[1, 0].item()}")
    print(f"  table[0] 整杯小明，形状 {tuple(table[0].shape)}  数值 {table[0].tolist()}")
    print(f"  table[:, 1] 两杯的冰，形状 {tuple(table[:, 1].shape)}  数值 {table[:, 1].tolist()}")

    print("\n== 同样 3 个数，轴的个数可以不同")
    one_row = torch.tensor([[8.0, 2.0, 6.0]])
    one_col = torch.tensor([[8.0], [2.0], [6.0]])
    show("一排 (3,)", person)
    show("一行的表 (1, 3)", one_row)
    show("一列的表 (3, 1)", one_col)

    print("\n== 线性层按最后一轴当作原料")
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.0]]))
        layer.bias.copy_(torch.tensor([1.0, 0.0]))
    print(f"  一排 (3,) 进去，出来形状 {tuple(layer(person).shape)}  数值 {layer(person).tolist()}")
    print(f"  一行 (1, 3) 进去，出来形状 {tuple(layer(one_row).shape)}  数值 {layer(one_row).tolist()}")
    print(f"  两杯 (2, 3) 进去，出来形状 {tuple(layer(table).shape)}  数值 {layer(table).tolist()}")

    print("\n== 形状 (2,) 的一排数，会加到每一行的最后一轴上")
    weighted = torch.tensor([[5.0, 8.0], [-5.0, 0.0]])
    bias = torch.tensor([1.0, 0.0])
    added = weighted + bias
    show("乘完权重、还没加底分", weighted)
    show("b1", bias)
    show("每一行都加上 [1, 0]", added)

    print("\n== 内存顺序：从左到右、再下一行。view 按这个顺序重新切")
    flat = table.view(6)
    regrouped = table.view(3, 2)
    swapped = table.t()
    show("按行拉直 (6,)", flat)
    show("切成 (3, 2)", regrouped)
    show("转置 t() 得到 (3, 2)", swapped)

    print("\n== 一张小图：轴是 通道、行、列")
    picture = torch.tensor([
        [
            [0.0, 8.0, 0.0],
            [0.0, 8.0, 0.0],
        ]
    ])
    show("一笔竖线", picture)
    print(f"  picture[0, 0, 1] 第 0 通道、第 0 行、第 1 列 = {picture[0, 0, 1].item()}")
    print(f"  picture[0, 1, 1] 第 0 通道、第 1 行、第 1 列 = {picture[0, 1, 1].item()}")
    print(f"  拉直后 {picture.view(1, -1).tolist()}")

    print("\n== 两张小图叠在一起：最前面多一个「第几张」的轴")
    blank = torch.zeros(1, 2, 3)
    batch = torch.stack([picture, blank])
    show("一批", batch)
    print(f"  batch[0, 0, 0, 1] 第 0 张、通道 0、行 0、列 1 = {batch[0, 0, 0, 1].item()}")
    print(f"  batch[1, 0, 0, 1] 第 1 张的同一格 = {batch[1, 0, 0, 1].item()}")
    print(f"  每张拉成 6 个数后形状 {tuple(batch.view(2, -1).shape)}")
    print(f"  数值 {batch.view(2, -1).tolist()}")

    print("\n== 类型和存放位置是整块数共有的，不随下标变化")
    label = torch.tensor([0])
    print(f"  小明 dtype {person.dtype}  device {person.device}")
    print(f"  标签 dtype {label.dtype}  数值 {label.tolist()}  形状 {tuple(label.shape)}")


if __name__ == "__main__":
    main()
