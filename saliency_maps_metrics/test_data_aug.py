import sys
import torch
import torchvision 

from multi_step_metrics import DAUC, IAUC
from single_step_metrics import AD, ADD

def test_data_aug():

    metric_dic = {"DAUC":DAUC, "IAUC":IAUC, "AD":AD, "ADD":ADD}
    metric_list = list(metric_dic.keys())
    is_multi_step = {"DAUC":True, "IAUC":True, "AD":False, "ADD":False}

    func_list = ["black","blur","black","blur"]

    data = torch.zeros(1,1,224,224)
    x = torch.arange(data.shape[3]).unsqueeze(0)
    y = torch.arange(data.shape[2]).unsqueeze(1)
    data = (x % 2==0)*(y%2==0)
    data = data.expand(4,3,-1,-1).float()

    patch_res = 14

    cent_x,cent_y = patch_res//2,patch_res//2
    var_x,var_y = 2,2

    x = torch.arange(patch_res).unsqueeze(0)
    y = torch.arange(patch_res).unsqueeze(1)

    expl = torch.exp(-((x-cent_x)**2)/var_x-((y-cent_y)**2)/var_y)
    expl = (expl - expl.min())/(expl.max() - expl.min())
    expl = expl.unsqueeze(0).unsqueeze(0)
    expl = expl.expand(data.shape[0],-1,-1,-1)

    data_masked_list = []

    for i in range(len(data)):
        metric_ind = i
        metric_name = metric_list[metric_ind]

        if is_multi_step[metric_name]:
            metric = metric_dic[metric_name](data.shape,expl.shape,func_list[i],True)

            data_to_replace_with_i = metric.init_data_to_replace_with(data[i:i+1])
            data_i = metric.preprocess_data(data[i:i+1]) 
            expl_i = expl[i:i+1]

            k = expl.shape[2]*expl.shape[3]//4
       
            mask,_ = metric.compute_mask(expl_i,data.shape,k)
       
            data_masked = metric.apply_mask(data_i,data_to_replace_with_i,mask)
        else:
            metric = metric_dic[metric_name](func_list[i])
            mask = metric.compute_mask(expl[i:i+1],data.shape)

            data_to_replace_with = metric.init_data_to_replace_with(data[i:i+1])
            data_masked = metric.apply_mask(data[i:i+1],data_to_replace_with,mask)

        data_masked_list.append(data_masked)
    
    data_masked_list = torch.cat(data_masked_list,dim=0)
    torchvision.utils.save_image(data_masked_list,"../data_aug_test.png")

if __name__ == "__main__":
    test_data_aug()