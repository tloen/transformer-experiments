from torch._dynamo import optimize
import torch


def fn(x, y):
    a = torch.cos(x).cuda()
    b = torch.sin(y).cuda()
    return a + b


new_fn = optimize("nvprims_aten")(fn)
# new_fn = fn
input_tensor = torch.randn(10000).to(device="cuda:0")
a = new_fn(input_tensor, input_tensor)
print(a)
