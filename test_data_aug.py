import sys
import torch
import torchvision 

from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion
from saliency_maps_metrics.single_step_metrics import IIC_AD, ADD

def test_data_aug():

    metric_dic = {"Deletion":Deletion, "Insertion":Insertion, "AD":IIC_AD, "ADD":ADD}
    metric_list = list(metric_dic.keys())
    metric_list = metric_list *  2
    is_multi_step = {"Deletion":True, "Insertion":True, "AD":False, "ADD":False}

    func_list = ["black","blur","black","blur","black","blur","black","blur"]

    data = torch.zeros(1,1,224,224)
    x = torch.arange(data.shape[3]).unsqueeze(0)
    y = torch.arange(data.shape[2]).unsqueeze(1)
    data = (x % 2==0)*(y%2==0)
    data = data.expand(len(metric_list),3,-1,-1).float()

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

    for i in range(len(metric_list)):

        metric_ind = i
        metric_name = metric_list[metric_ind]

        data_i = data[i:i+1]
        expl_i = expl[i:i+1]

        if metric_name == "Insertion":
            metric = metric_dic[metric_name](func_list[i],cumulative=False)
        else:
            metric = metric_dic[metric_name](func_list[i])

        if i >= len(metric_list)//2:
            masking_data_i = torch.zeros_like(data_i)
            masking_data_i[:,0] = 1
        else:
            masking_data_i = metric.get_masking_data(data_i)

        if is_multi_step[metric_name]:
         
            dic = metric.choose_data_order(data_i,masking_data_i)
            data1_i,data2_i = dic["data1"],dic["data2"]

            k = expl.shape[2]*expl.shape[3]//4
       
            total_pixel_nb = expl.shape[2]*expl.shape[3]
            step_nb = min(metric.max_step_nb,total_pixel_nb) if metric.bound_max_step else total_pixel_nb
            pixel_removed_per_step = total_pixel_nb//step_nb

            mask,_ = metric.compute_mask(expl_i,data.shape,k,pixel_removed_per_step)
       
            data_masked = metric.apply_mask(data1_i,data2_i,mask)

        else:
            mask = metric.compute_mask(expl_i,data.shape)

            data_masked = metric.apply_mask(data_i,masking_data_i,mask)

        data_masked_list.append(data_masked)
    
    data_masked_list = torch.cat(data_masked_list,dim=0)
    torchvision.utils.save_image(data_masked_list,"data_aug_test.png")

if __name__ == "__main__":
    test_data_aug()