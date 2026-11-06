import torch

dtype = torch.float
device = torch.device("cpu")
#a = torch.randn(2,3,device=device,dtype=dtype)
#b = torch.randn(2,3,device=device,dtype=dtype)
a = torch.tensor([[1,2,3],[4,5,6]],dtype=dtype)
b = torch.tensor([[1,2,3],[4,5,6]],dtype=dtype)
print("张量 a:")
print(a)

print("张量 b:")
print(b)


print("a 和 b 的逐元素乘积")
print(a*b)

print(a+b)
print(a-b)
print(a/b)


print(torch.matmul(a.T,b))


print("torch.mean(a): " )
print(torch.mean(a))

print("张量 a 所有元素的总和")
print(a.sum(dim=1))

print("张量 a 中最大值：")
print(a.max())

c = torch.tensor([[[ 1.0,  2.0,  3.0,  4.0],[ 5.0,  6.0,  7.0,  8.0],[ 9.0, 10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0, 16.0],[17.0, 18.0, 19.0, 20.0],[21.0, 22.0, 23.0, 24.0]]],dtype=dtype)
print(c.sum(dim=0))


if torch.cuda.is_available() :
    print("GPU") 
else: 
    print("CPU")

x = torch.linspace(0, 5, 5)
print(x)

y = torch.eye(5,5)
print(y)


print("c 的 shape:",c.shape)


print("最大值的索引",c.argmax(dim=0))
print("最大值计算",c.softmax(dim=1))
print("求均值",c.mean(dim=1))
