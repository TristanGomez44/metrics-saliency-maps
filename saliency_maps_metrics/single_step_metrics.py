import numpy as np
import torch
import torch.functional as F
import sys 
from .data_replace import select_data_replace_method

def min_max_norm(arr):

    if arr.min() == arr.max():
        return arr 
    else:
        return (arr-arr.min())/(arr.max()-arr.min())

class SingleStepMetric():

    def __init__(self,data_replace_method="black") -> None:
        
        self.data_replace_func = select_data_replace_method(data_replace_method)

    def get_masking_data(self,data):
        return self.data_replace_func(data)

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

    def compute_scores(self,model,data,explanations,class_to_explain_list=None,data_to_replace_with=None):
        masks = self.compute_mask(explanations,data.shape).to(data.device)
        if data_to_replace_with is None:
            data_to_replace_with = self.get_masking_data(data)
        data_masked = self.apply_mask(data,data_to_replace_with,masks)

        score_list = []
        score_masked_list = []
        for i in range(len(data)):
            if class_to_explain_list is None:
                class_to_explain = torch.argmax(model(data[i:i+1]),axis=1)[0]
            else:
                class_to_explain = class_to_explain_list[i]
        
            score = model(data[i:i+1])[0,class_to_explain]
            score_masked = model(data_masked[i:i+1])[0,class_to_explain].item()          
            score_list.append(score)
            score_masked_list.append(score_masked)

        score_list = torch.tensor(score_list)
        score_masked_list = torch.tensor(score_masked_list)

        return score_list,score_masked_list

    def __call__(self,model,data,explanations,class_to_explain_list,data_to_replace_with=None):
        score_list,score_masked_list = self.compute_scores(model,data,explanations,class_to_explain_list,data_to_replace_with)
        return self.compute_metric(score_list,score_masked_list)

    def compute_metric(self,score,score_masked):
        raise NotImplementedError

class IIC_AD(SingleStepMetric):

    def compute_metric(self,score,score_masked):
        return {"iic":(score_masked > score).float().mean(),"ad":(torch.clamp(score-score_masked,min=0)/score).mean()}

class ADD(SingleStepMetric):

    def preprocess_mask(self,masks):
        return 1-masks

    def compute_metric(self,score,score_masked):
        return {"add":((score-score_masked)/score).mean()}