# PyTorch：读数字项目里的那些对象

`examples/mnist-digits/train.py` 是一条训练。PyTorch 把这条训练拆成几类对象：张量、模型、数据批、损失、优化器。这些对象各自对应人工神经网络里的哪一格，先看 [基础框架](pytorch-foundations.md)。下面按 `train.py` 真正调用的顺序说明。手算用奶茶网络里的小明 `[8, 2, 6]`，和 [框架笔记](00-neural-network-framework.md) 同一组权重。读数字项目只是把 3 个数换成 784 个亮度，把 2 个分数换成 10 个分数。

可运行的对照脚本是 [pytorch_tour.py](../../examples/mnist-digits/pytorch_tour.py)。在 `examples/mnist-digits` 里执行 `python pytorch_tour.py`，打印出来的数和下面相同。它不下载 MNIST。

<a id="pieces"></a>

## 一条训练用到哪些东西

| `train.py` 里的名字 | PyTorch 里的类 | 这一步留下什么 |
| --- | --- | --- |
| `images`、`labels`、`scores` | `torch.Tensor` | 带形状的一张数字表 |
| `DigitClassifier` | `nn.Module` | 层，以及层里的 `W`、`b` |
| `nn.Linear` | 一层线性层 | `weight` 和 `bias` 两个张量 |
| `datasets.MNIST` | `Dataset` | 按编号取出一张图和一个标签 |
| `DataLoader` | 批处理 | 一次取出 64 张，拼成一个张量 |
| `transforms.ToTensor` | 变换 | 亮度从 0～255 收到 0～1 |
| `nn.CrossEntropyLoss` | 损失 | 一个标量：正确那一类有多不准 |
| `loss.backward()` | 自动求导 | 每个 `W`、`b` 旁边放上同形状的梯度 |
| `torch.optim.Adam` | 优化器 | 按梯度改 `W`、`b`，并记住每个参数自己的步子 |
| `model.train()` / `model.eval()` | 模式开关 | 这个网络里两次前向的算式相同 |
| `torch.no_grad()` | 关掉记录 | 测试时不保存“怎么算出来的” |
| `torch.manual_seed(0)` | 随机数种子 | 新建线性层时抽到同一套初始权重 |
| `state_dict` | 参数字典 | 训练结束后写到 `digit.pt` 的就是这份字典 |

torchvision 是配套库，提供 MNIST 和 `ToTensor`。算式、求导、改权重都在 `torch` 里。

<a id="tensor"></a>

## 张量

张量是一块排好轴的数。取一格要写的下标个数，等于轴的个数。小明写成 `[[8, 2, 6]]` 时形状是 `(1, 3)`：外面那一轴长度 1，表示这一批只有一杯；里面那一轴是甜、冰、茶。两杯叠成表是 `(2, 3)`，`table[0, 1]` 是小明的冰，等于 2。一批数字图是 `(64, 1, 28, 28)`，四个轴依次是第几张、通道、行、列。

下标、`(3,)` 和 `(1, 3)` 的差别、`view` 与转置，按格写在 [pytorch-tensor.md](pytorch-tensor.md)。可运行的对照是 `examples/mnist-digits/tensor_tour.py`。

<a id="module"></a>

## 模型：Module 和 Linear

`DigitClassifier` 继承 `nn.Module`。两件套：

- `__init__` 里创建层。写成 `self.hidden = nn.Linear(...)` 之后，这一层的权重登记进模型。
- `forward` 里写前向算式。调用 `model(drink)` 时，PyTorch 先做模块自己的登记工作，再进入 `forward`。

奶茶网络和读数字网络是同一个类结构，只是宽度不同：

```python
class TeaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 2)    # 3 个原料 → 2 个检测
        self.output = nn.Linear(2, 2)    # 2 个检测 → 喜欢、不喜欢

    def forward(self, drinks):
        detected = torch.relu(self.hidden(drinks))
        return self.output(detected)
```

<a id="class"></a>

## `class SimpleNN(nn.Module)`

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
y = model(x)
```

`class SimpleNN(nn.Module)` 定义一种新的模型。这一行拆开是：

```text
class      开始定义类
SimpleNN   类名
(nn.Module) 父类，写在括号里
:          类的内容从下一行缩进开始
```

括号里的 `nn.Module` 就是继承。`SimpleNN` 的父类名单是 `Module`，一个 `SimpleNN()` 实例同时也是 `nn.Module`。查找方法时先看 `SimpleNN`，没有再看 `Module`，再看 `object`。

Java 里同一句写成 `class SimpleNN extends Module {`。`extends` 标出父类，大括号包住类的内容。Python 把父类放进类名后面的括号，用冒号和缩进代替大括号。Python 的括号里可以写多个父类，例如 `class C(A, B)`。Java 的 `extends` 后面是一个类，额外的接口用 `implements` 另写。

构造也换了写法。Java 的构造方法和类同名，`this` 可以不写出来。Python 的构造方法固定叫 `__init__(self)`，`self` 要写在参数列表里，它就是这份对象。调用父类构造，Java 写 `super();`，这里写 `super(SimpleNN, self).__init__()`。

`SimpleNN` 因此带上父类已经写好的能力：登记参数、`train()`、`eval()`、`state_dict()`，以及「把实例当函数调用」。

`__init__` 在 `SimpleNN()` 创建实例时执行一次。`super(SimpleNN, self).__init__()` 先让父类把自己的登记表准备好。后面两行 `self.fc1 = ...`、`self.fc2 = ...` 才把两个线性层写进这张表。`fc1`、`fc2` 只是自己起的名字。读数字模型里的 `self.hidden`、`self.output` 是同一写法。

`def forward(self, x)` 定义前向方法。

- `def` 开始一个方法。
- `self` 是这一份模型。`self.fc1` 就是上面登记的第一层。
- `x` 是调用时传进来的输入张量。形状是 `(批量, 2)`，因为 `fc1` 要 2 个特征。
- 方法里的 `x = ...` 用同一个名字接住上一层的输出。第一行之后，`x` 是 ReLU 之后的两个检测；第二行之后，`x` 是一个预测值。
- `return x` 把最后这一排数交回调用者。这就是网络的输出。

使用分两步。先 `model = SimpleNN()`，权重在这一步建出来，一份模型可以反复用。再 `y = model(x)`。这个括号调用走的是父类的 `__call__`，它再进入你写的 `forward`。`model.forward(x)` 直接执行同一个方法，数值相同。训练循环里写的是 `model(x)`。

`model(features)` 里的张量不是交给构造方法的。构造方法是 `__init__(self)`，在 `model = SimpleNN()` 时已经执行完，参数列表里只有 `self`。后面这句是对已经建好的对象再调用一次。父类 `Module` 写了 `__call__`，所以对象名后面可以加括号。括号里的 `features` 被原样传进 `forward` 的 `x`。`fc1` 要求 `x` 的最后一轴长度是 2，所以这里的 `features` 形状是 `(2, 2)`：两行样本，每行两个特征。

`by_call` 的形状是 `(2, 1)`，两行各一个预测值，排成 `[[2.0], [0.0]]`。`view(-1)` 按元素总数把形状收成一根轴，`2×1 = 2`，得到 `[2.0, 0.0]`。`tolist()` 把张量抄成 Python 的列表。`print("model(x)", ...)` 里的 `"model(x)"` 只是打印在数字前面的标签。

把系数填成可以手算的数：`fc1` 的两行是 `[0, 1]` 和 `[1, 0]`，底分 `[1, 0]`；`fc2` 的一行是 `[0.4, 0.1]`，底分 `0`。输入两行 `[[8, 2], [0, −6]]`：

```text
第一行  8, 2
  检测 = [0×8 + 1×2 + 1,  1×8 + 0×2 + 0] = [3, 8]
  ReLU 之后仍是 [3, 8]
  输出 = 0.4×3 + 0.1×8 + 0 = 2

第二行  0, −6
  检测 = [0×0 + 1×(−6) + 1,  1×0 + 0×(−6) + 0] = [−5, 0]
  ReLU 之后是 [0, 0]
  输出 = 0
```

输出形状是 `(2, 1)`。`print(model)` 打印的是结构，不是这些系数：

```text
SimpleNN(
  (fc1): Linear(in_features=2, out_features=2, bias=True)
  (fc2): Linear(in_features=2, out_features=1, bias=True)
)
```

参数名是 `fc1.weight`、`fc1.bias`、`fc2.weight`、`fc2.bias`。对照脚本是 [simple_nn.py](../../examples/pytorch/lesson2/simple_nn.py)。

`named_parameters()` 列出要训练的张量。奶茶网络是四份：

| 名字 | 形状 | 对应笔记 |
| --- | --- | --- |
| `hidden.weight` | `(2, 3)` | 笔记里的 `W1` 是 3×2，这里存的是它的转置 |
| `hidden.bias` | `(2,)` | `b1 = [1, 0]` |
| `output.weight` | `(2, 2)` | 笔记里的 `W2` 是 2×2，这里同样是转置后的存法 |
| `output.bias` | `(2,)` | `b2 = [0, 0.5]` |

每一份在创建时 `requires_grad` 都是 `True`：后面的 `backward` 会给它填梯度。

笔记的乘法是 `y = x W + b`。PyTorch 的线性层是 `y = x @ weight.T + bias`。把笔记里的 `W1` 转置后抄进 `weight`，乘积相同。小明：

```text
清爽 = 0×8 + 1×2 + 0.5×6 + 1 = 6
甜腻 = 1×8 + 0×2 + 0×6   + 0 = 8
ReLU 之后仍是 [6, 8]

喜欢   = 0.4×6 + 0.1×8 + 0   = 3.2
不喜欢 = −0.2×6 + 0×8 + 0.5 = −0.7
```

`model(drink)` 打出来就是 `[3.2, -0.7]`。

新建 `nn.Linear` 时，权重按 Kaiming uniform 抽随机数，偏置在 `±1/√输入维数` 里均匀抽。读数字第一层的输入维数是 784，偏置的抽取范围是 `±1/√784 ≈ ±0.0357`。`torch.manual_seed(0)` 放在创建模型之前，两次运行抽到同一套初始值。脚本里种子为 0 时，一个 `Linear(3, 2)` 的第一格是 `-0.004323`。

读数字模型 `DigitClassifier()` 在种子 0 下有 **101770** 个参数：`128×784 + 128 + 10×128 + 10`。`state_dict()` 的四个名字和奶茶网络一样：`hidden.weight`、`hidden.bias`、`output.weight`、`output.bias`。两张全黑的图送进去，分数的形状是 `(2, 10)`。

<a id="data"></a>

## 数据集和一批

`Dataset` 负责“第 i 张是什么”。MNIST 的 `__getitem__` 返回一张图和一个整数标签。`ToTensor` 在取出时把 0～255 的亮度除以 255，并排成 `(1, 28, 28)`：通道在前，然后是高、宽。

`DataLoader` 负责“一次拿多少张，怎么拼”。`batch_size=2`、三杯饮料、`shuffle=False` 时：

```text
第 1 批  x (2, 3)  y [0, 1]     小明、热浓茶
第 2 批  x (1, 3)  y [0]       小红，最后一批可以更短
```

训练集 60000 张、`batch_size=64` 时，一轮有 938 批：937 批是 64 张，最后一批是 32 张。`shuffle=True` 让每一轮的顺序重新打乱，权重仍是同一套。测试集 `shuffle=False`，所以“第一张测试图”每次都是同一张。

一批里的 64 张共用 `W` 和 `b`。线性层把这 64 行一次乘完，得到 `(64, 10)` 的分数。

<a id="loss"></a>

## 损失

`nn.CrossEntropyLoss` 做两件事：把一行分数变成加总为 1 的百分比，再取正确那一格的负对数。小明的分数 `[3.2, -0.7]`、标签 `0`：

```text
百分比 ≈ [0.980160, 0.019840]，加总 1
损失   = −ln(0.980160) ≈ 0.020040
```

`loss` 自己仍是一个张量。`loss.item()` 把它抄成 Python 的 `float`，训练循环里累加的是这个普通数字。

热浓茶 `[0, -6, 2]` 的清爽是 −4，ReLU 之后清爽和甜腻都是 0，分数是 `[0, 0.5]`，标签是不喜欢（1），损失约 0.474。两杯一起送进损失时，默认对两杯取平均：`(0.020040 + 0.474077) / 2 ≈ 0.247058`。读数字的一批是 64 张，损失是这 64 张的平均。

模型的 `forward` 返回分数。百分比在损失内部算。打印给人看时，`train.py` 再用 `torch.softmax(scores, dim=1)` 把每一行变成百分比。`dim=1` 表示在 10 个数字之间归一化，一行加起来是 1。

<a id="backward"></a>

## backward：梯度

`loss.backward()` 从损失往回走，给每个 `requires_grad=True` 的参数填上 `.grad`。梯度的形状和参数相同。它的含义是：这个参数单独加大一点，损失会加大多少。

小明这一杯、标签是喜欢时，`output.bias.grad` 是 `[-0.019840, 0.019840]`。可以手算。一批只有一杯时，底分的梯度等于「当前百分比 − 答案」：

```text
喜欢这一格的答案是 1    梯度 = 0.980160 − 1 = −0.019840
不喜欢这一格的答案是 0  梯度 = 0.019840 − 0 =  0.019840
```

喜欢的梯度是负的：把喜欢的底分加大，损失会下降。不喜欢的梯度是正的：把不喜欢的底分加大，损失会上升。训练时要朝损失下降的方向改，所以下一步会加上梯度的相反数。

第一层权重还要再乘回去。喜欢、不喜欢对清爽的系数是 0.4 和 −0.2，清爽这一路的损失梯度是：

```text
−0.019840×0.4 + 0.019840×(−0.2) = −0.011904
```

清爽对甜、冰、茶的系数是 0、1、0.5，输入是 8、2、6。`hidden.weight` 第一行（清爽）的梯度是：

```text
−0.011904 × [8, 2, 6] ≈ [−0.095232, −0.023808, −0.071424]
```

脚本打印的是 `[-0.095233, -0.023808, -0.071425]`，差在百分比的四舍五入。甜腻那一行同样是「甜腻的梯度 × [8, 2, 6]」。6 和 8 都大于 0，ReLU 把梯度原样送回第一层。热浓茶那种清爽为负的杯子，ReLU 之后是 0，这一路的梯度也是 0。

<a id="step"></a>

## zero_grad 和 step

梯度会留在参数上。再调用一次 `backward`，新的梯度加到旧的上面。一个数可以看清这件事。

设 `w = 1`，`分数 = 3w`，`损失 = (分数 − 5)²`。

```text
分数 = 3
损失 = (3 − 5)² = 4
d(损失)/d(分数) = 2×(3 − 5) = −4
d(损失)/dw     = −4 × 3 = −12
```

`2×(3 − 5)` 是损失对分数的变化率，后面的 `3` 是分数对 `w` 的变化率。两步各自怎么量出来，写在 [pytorch-optimizer.md](pytorch-optimizer.md)。

`torch.optim.SGD([w], lr=0.01)` 没传动量和权重衰减，一步是：

```text
w ← w − 0.01 × (−12) = 1.12
```

分数从 3 变成 `3×1.12 = 3.36`，靠近目标 5。连续两次 `backward`、中间不把梯度清掉，梯度从 −12 变成 −24。`train.py` 里每一批开头的 `optimizer.zero_grad()` 就是把上一批留下的梯度清成 0，让这一批的梯度单独决定这一步。

读数字项目用的是 `torch.optim.Adam(model.parameters(), lr=1e-3)`。`model.parameters()` 把四个参数张量交给 Adam。PyTorch 这一版 Adam 的默认值是 `betas=(0.9, 0.999)`、`eps=1e-8`。同一个 `w = 1`、梯度 −12、学习率 0.01，Adam 的第一步可以手算：

```text
m = 0.1 × (−12) = −1.2          梯度的滑动平均，第一步只有 10% 的当前梯度
v = 0.001 × (−12)² = 0.144      梯度平方的滑动平均
m̂ = −1.2 / (1 − 0.9) = −12      把“第一步被 0.9 缩小”补回来
v̂ = 0.144 / (1 − 0.999) = 144
w ← 1 − 0.01 × (−12) / √144 = 1.01
```

第一步在这个单参数例子里，Adam 把 `w` 挪动一个学习率的距离，方向仍是让损失变小。SGD 用原始梯度的大小，同一步挪到 1.12。从第二步起，Adam 按每个参数自己的梯度历史放大或缩小步子。论文里的 Adam 用 `β2 = 0.98`、`ε = 1e-9`。这个项目没有把这两项传进去，用的是 PyTorch 的默认值。

一轮的四行，对应上面这三节：

```python
optimizer.zero_grad()          # 清掉上一批的梯度
scores = model(images)         # 前向，记下算式
loss = loss_fn(scores, labels) # 一个标量
loss.backward()                # 填 .grad
optimizer.step()               # 改 W、b
```

<a id="eval"></a>

## train、eval 和 no_grad

`model.eval()` 把 `model.training` 设成 `False`，并同样设到每一层，比如 `model.hidden.training`。`model.train()` 设回 `True`。这一下不改 `weight` 和 `bias`，也不关掉梯度记录。评估模式下线性层仍是 `2×3 + 0 = 6`，`grad_fn` 仍是 `AddmmBackward0`。不记录梯度的是包在 `evaluate` 外面的 `@torch.no_grad()`。

`DigitClassifier` 只有线性层和 ReLU。同一张全 0 的图，`train()` 和 `eval()` 打出的 10 个分数相同。`train.py` 仍在训练循环里调用 `model.train()`，在 `evaluate` 里调用 `model.eval()`。这个网络的前向算式不看这个开关。

会读这个开关的层，两种模式用不同的算式。`Dropout(p=0.5)` 收到 `[2, 4, 6, 8]`，种子是 0：

```text
train  [0, 0, 12, 0]
eval   [2, 4, 6, 8]
```

`train()` 按 0.5 的概率把一格清成 0，留下来的那格乘 `1 / (1 − 0.5) = 2`，所以 6 变成 12。`eval()` 把四个数原样送出，不再随机清零。

`BatchNorm1d` 收到一批两个数 `1` 和 `3`，缩放是 1，底分是 0。`train()` 用这一批自己的均值 2、方差 1：

```text
(1 − 2) / √(1 + 1e-5) = −0.999995
(3 − 2) / √(1 + 1e-5) =  0.999995
```

同时把累计均值更新成 `0.1×2 = 0.2`，累计方差更新成 `0.9×1 + 0.1×2 = 1.1`。`eval()` 不再用这一批的均值，也不再改累计值，用已经记下的 0.2 和 1.1：

```text
(1 − 0.2) / √(1.1 + 1e-5) = 0.762767
(3 − 0.2) / √(1.1 + 1e-5) = 2.669683
```

所以 `eval()` 的作用是：让 Dropout 停止随机清零，让 BatchNorm 改用累计的均值和方差，并且测试的这一批不再写进那份累计。这个读数字网络没有这两层，调用之后分数不变。

`@torch.no_grad()` 包住 `evaluate`。里面的乘法不记录 `grad_fn`。脚本里同一杯：

```text
eval 且 no_grad    分数的 grad_fn 是 None
train              分数的 grad_fn 是 AddmmBackward0
```

`AddmmBackward0` 就是线性层那次矩阵乘留下来的回传入口。测试不调用 `backward`，所以不需要这条入口。`show_one_prediction` 同样包在 `no_grad` 里。它用 `scores.argmax(dim=1)` 取每一行分数最大的下标，那个下标就是读出的数字。

<a id="save"></a>

## 把权重存下来

`model.state_dict()` 是一个字典，键就是上面四个名字，值是对应的张量。`train.py` 训练结束后执行：

```python
torch.save(model.state_dict(), "digit.pt")
```

文件里是学到的 `W` 和 `b`，没有把 Python 类一起写进去。`predict.py` 先 `DigitClassifier()` 建一个同样结构的空模型，再 `load_state_dict(...)` 把数抄回去。`hidden_size` 要和训练时相同，否则 `(128, 784)` 对不上。`map_location="cpu"` 保证加载后的张量在 CPU 上。`weights_only=True` 只读张量。`digit.pt` 在 `.gitignore` 里，不进 git。

<a id="check"></a>

## 过关自测

1. 小明的张量形状为什么是 `(1, 3)`，一批 64 张数字图拉直后为什么是 `(64, 784)`？
2. 笔记里的 `W1` 是 3×2。`hidden.weight` 的形状为什么是 `(2, 3)`？小明的分数仍是 `[3.2, -0.7]`，哪一步把转置乘了回去？
3. 喜欢的底分梯度是 −0.019840。把这个底分加大一点点，损失往哪边走？
4. 同一批上 `backward` 两次、中间没有 `zero_grad`，梯度会变成多大？
5. 这个项目的 Adam 用的 `β2` 是 0.999。论文里写的是哪一个？
