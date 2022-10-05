import numpy as np
import torch
import torch.functional as F
from utils import min_max_norm

class SingleStepMetric():

    def __call__(self,model,data,explanations,class_to_explain):

        explanations = min_max_norm(explanations)
        explanations = torch.nn.functional.interpolate(explanations,size=(data.shape[-1]),mode="bicubic").to(data.device)                    
        
        explanations = self.preprocess_expl(explanations)
        
        data_masked = data*explanations

        sample_list = []
        for i in range(len(data)):
            score = model(data[i:i+1])[0,class_to_explain]
            score_masked = model(data_masked[i:i+1])[0,class_to_explain]            
            sample_list.append(self.compute_metric_sample(score,score_masked))
        sample_list = torch.cat(sample_list,dim=0).float().mean()

        return sample_list

    def preprocess_expl(self,explanations):
        return explanations

    def compute_metric_sample(self,score,score_masked):
        raise NotImplementedError

class IIC(SingleStepMetric):

    def compute_metric_sample(self,score,score_masked):
        return score_masked > score

class AD(SingleStepMetric):

    def compute_metric_sample(self,score,score_masked):
        return torch.clamp(score-score_masked,min=0)/score  

class ADD(SingleStepMetric):

    def preprocess_expl(self,explanations):
        return 1-explanations

    def compute_metric_sample(self,score,score_masked):
        return (score-score_masked)/score  