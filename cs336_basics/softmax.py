import torch

def softmax(x,dim):
    x = x - torch.max(x,dim=dim,keepdim=True)[0]
    exp_p = torch.exp(x)
    p = exp_p/torch.sum(exp_p,dim=dim,keepdim=True)

    return p