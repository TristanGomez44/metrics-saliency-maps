import numpy as np
import torch
import torch.functional as F
import sys 
from data_transf import select_data_transf

def min_max_norm(arr):

    if arr.min() == arr.max():
        return arr 
    else:
        return (arr-arr.min())/(arr.max()-arr.min())

class SingleStepMetric():

    def __init__(self,data_transf_str="black") -> None:
        
        self.data_transf_func = select_data_transf(data_transf_str)

    def init_data_to_replace_with(self,data):
        return self.data_transf_func(data)

    def preprocess_mask(self,masks):
        return masks

    def compute_mask(self,explanations,data_shape):
        masks = min_max_norm(explanations)
        masks = torch.nn.functional.interpolate(masks,size=(data_shape[2:]),align_corners=False,mode="bicubic")                       
        masks = self.preprocess_mask(masks)
        return masks

    def apply_mask(self,data,data_to_replace_with,mask):
        data_masked = data*mask + data_to_replace_with*(1-mask)
        return data_masked

    def __call__(self,model,data,explanations,class_to_explain):

        masks = self.compute_mask(explanations,data.shape).to(data.device)
        data_to_replace_with = self.init_data_to_replace_with(data)
        data_masked = self.apply_mask(data,data_to_replace_with,masks)

        sample_list = []
        for i in range(len(data)):
            score = model(data[i:i+1])[0,class_to_explain]
            score_masked = model(data_masked[i:i+1])[0,class_to_explain]            
            sample_list.append(self.compute_metric_sample(score,score_masked))
        mean_value = torch.cat(sample_list,dim=0).float().mean()

        return mean_value

    def compute_metric_sample(self,score,score_masked):
        raise NotImplementedError

class IIC(SingleStepMetric):

    def compute_metric_sample(self,score,score_masked):
        return score_masked > score

class AD(SingleStepMetric):

    def compute_metric_sample(self,score,score_masked):
        return torch.clamp(score-score_masked,min=0)/score  

class ADD(SingleStepMetric):

    def preprocess_mask(self,masks):
        return 1-masks

    def compute_metric_sample(self,score,score_masked):
        return (score-score_masked)/score  