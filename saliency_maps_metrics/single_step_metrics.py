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
        
        self.data_replace_method = data_replace_method
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

    def compute_scores(self,model,data,explanations,class_to_explain_list=None,data_to_replace_with=None,save_all_class_scores=False):
        
        with torch.no_grad():
            
            masks = self.compute_mask(explanations,data.shape).to(data.device)
            if data_to_replace_with is None:
                data_to_replace_with = self.get_masking_data(data)
            data_masked = self.apply_mask(data,data_to_replace_with,masks)

            score_list = []
            score_masked_list = []

            if class_to_explain_list is None:
                class_to_explain = torch.argmax(model(data),axis=1)
            else:
                class_to_explain = class_to_explain_list

            batch_inds = np.arange(len(data))

            score_list = model(data)
            if not save_all_class_scores:
                score_list = score_list[batch_inds,class_to_explain]

            score_masked_list = model(data_masked)
            if not save_all_class_scores:
                score_masked_list = score_masked_list[batch_inds,class_to_explain]

            score_list = score_list.cpu().numpy()
            score_masked_list = score_masked_list.cpu().numpy()

        return score_list,score_masked_list

    def __call__(self,model,data,explanations,class_to_explain_list,data_to_replace_with=None):
        raise NotImplementedError


    def compute_metric(self,score,score_masked):
        raise NotImplementedError

class IIC_AD(SingleStepMetric):

    def __call__(self, model, data, explanations, class_to_explain_list, data_to_replace_with=None):
        score_list,score_masked_list = self.compute_scores(model,data,explanations,class_to_explain_list,data_to_replace_with)
        result_dic = self.compute_metric(score_list, score_masked_list)
        for metric_name in result_dic:
            result_dic[metric_name] = result_dic[metric_name].mean()
        return result_dic

    def compute_metric(self, score, score_masked):
        iic = (score_masked > score).astype("float")
        ad = (np.maximum(score-score_masked,0)/score)
        return {"iic":iic,"ad":ad}

class ADD(SingleStepMetric):

    def preprocess_mask(self,masks):
        return 1-masks

    def __call__(self, model, data, explanations, class_to_explain_list, data_to_replace_with=None):
        score_list,score_masked_list = self.compute_scores(model,data,explanations,class_to_explain_list,data_to_replace_with)
        result_dic = self.compute_metric(score_list, score_masked_list)
        for metric_name in result_dic:
            result_dic[metric_name] = result_dic[metric_name].mean()
        return result_dic

    def compute_metric(self, score, score_masked):
        add = ((score-score_masked)/score)
        return {"add":add}