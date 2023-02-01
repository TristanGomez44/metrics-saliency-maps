import numpy as np
import torch
import torch.functional as F
import sys 
from .data_transf import select_data_transf

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

        score_list = []
        score_masked_list = []
        for i in range(len(data)):
            score = model(data[i:i+1])[0,class_to_explain]
            score_masked = model(data_masked[i:i+1])[0,class_to_explain]            
            score_list.append(score)
            score_masked_list.append(score_masked)

        score_list = np.array(score_list)
        score_masked_list = np.array(score_masked_list)

        return self.compute_metric(score_list,score_masked_list)

    def compute_metric(self,score,score_masked):
        raise NotImplementedError

class IIC_AD(SingleStepMetric):

    def compute_metric(self,score,score_masked):
        return {"iic":(score_masked > score).mean(),"ad":(torch.clamp(score-score_masked,min=0)/score).mean()}

class ADD(SingleStepMetric):

    def preprocess_mask(self,masks):
        return 1-masks

    def compute_metric(self,score,score_masked):
        return {"add":((score-score_masked)/score).mean()}