import torch

# x 是用户直接创建的，属于叶子节点，grad_fn 为 None
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(x.grad_fn)  # 输出: None

# y 是通过加法运算计算出来的，它带有 grad_fn
y = x + 2
print(y.grad_fn)  # 输出: <AddBackward0 object at 0x...>