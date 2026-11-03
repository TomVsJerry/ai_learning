import torch

dtype = torch.float
device = torch.device("cpu")
#a = torch.randn(2,3,device=device,dtype=dtype)
#b = torch.randn(2,3,device=device,dtype=dtype)
a = torch.tensor([[1,2,3],[4,5,6]])
b = torch.tensor([[1,2,3],[4,5,6]])
print("张量 a:")
print(a)

print("张量 b:")
print(b)


print("a 和 b 的逐元素乘积")
print(a*b)

print("张量 a 所有元素的总和")
print(a.sum(dim=1))

print("张量 a 中最大值：")
print(a.max())

c = torch.tensor([[[ 1,  2,  3,  4],[ 5,  6,  7,  8],[ 9, 10, 11, 12]], [[13, 14, 15, 16],[17, 18, 19, 20],[21, 22, 23, 24]]])
print(c.sum(dim=0))