import torch 
import torch.nn.functional as F

def constant_val(data,value):
    return value*torch.ones_like(data,device=data.device)

def blur_data(data):
    kernel_size = max(data.shape[2]//5,9)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = torch.ones(kernel_size,kernel_size)
    kernel = kernel/kernel.numel()
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,1,-1,-1)
    kernel = kernel.to(data.device)
    data = F.conv2d(data,kernel,padding=kernel.size(-1)//2,groups=kernel.size(0))  
    return data

func_dic = {"black":lambda x:constant_val(x,0),"blur":blur_data}

def select_data_transf(transf_str):
    if transf_str in func_dic:
        transf = func_dic[transf_str]
    else:
        raise ValueError("Unkown data transformation",transf_str)
    return transf

