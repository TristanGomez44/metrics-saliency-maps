import sys
import numpy as np
import torch

from .data_transf import select_data_transf

def compute_correlation(score_var, all_sal_score_list):
    corr_list = []
    for i in range(len(score_var)):
        var_class_score = score_var[i][:,np.newaxis]
        sal_score = all_sal_score_list[i][:,np.newaxis]

        points = np.concatenate((var_class_score,sal_score),axis=-1)
        corr_list.append(np.corrcoef(points,rowvar=False)[0,1])

    corr_mean = np.array(corr_list).mean()
    return corr_mean

def compute_auc_metric(all_score_list):
    auc_list = []
    for i in range(len(all_score_list)):
        scores = all_score_list[i]
        auc = np.trapz(scores,np.arange(len(scores))/(len(scores)-1))
        auc_list.append(auc)

    auc_mean = np.array(auc_list).mean()
    return auc_mean

class AUC_Metric():

    def __init__(self,data_shape,explanation_shape,data_transf_str,bound_max_step=True):

        #Set this to true to limit the maximum number of inferences computed by the metric
        #The DAUC/IAUC protocol requires one inference per pixel of the explanation map.
        #This can be prohibitive for high resolution maps like Smooth-Grad.
        #In the case of a high-resolution map (>14x14), setting this arg to True results 
        #in one inference for every few pixels removed, instead of one inference per pixel.
        self.bound_max_step = bound_max_step 

        self.size_ratio = data_shape[3]//explanation_shape[3]
        self.total_pixel_nb = explanation_shape[2]*explanation_shape[3]
        self.step_nb = min(14*14,self.total_pixel_nb) if self.bound_max_step else self.total_pixel_nb
        self.pixel_removed_per_step = self.total_pixel_nb//self.step_nb

        self.data_transf_func = select_data_transf(data_transf_str)

    def init_data_to_replace_with(data):
        raise NotImplementedError

    def preprocess_data(data):
        raise NotImplementedError

    def compute_mask(self,explanation,data_shape,k):
        mask_flat = torch.ones(explanation.shape[2]*explanation.shape[3]).to(explanation.device)
        saliency_scores,inds = torch.topk(explanation.view(-1),k,0,sorted=True)
        mask_flat[inds] = 0
        mask = mask_flat.reshape(1,1,explanation.shape[2],explanation.shape[3])
        mask = torch.nn.functional.interpolate(mask,data_shape[2:],mode="nearest")
        return mask,saliency_scores

    def apply_mask(self,data,data_to_replace_with,mask):
        data_masked = data*mask + data_to_replace_with*(1-mask)
        return data_masked

    def update_data(self,data, i, y1, y2, x1, x2,data_to_replace_with):
        data[i:i+1,:,y1:y2,x1:x2] = data_to_replace_with[i:i+1,:,y1:y2,x1:x2]
        return data

    def compute_calibration_metric(all_score_list, all_sal_score_list):
        raise NotImplementedError

    def make_result_dic(self,auc_metric,calibration_metric):
        raise NotImplementedError

    def __call__(self,model,data,explanations,class_to_explain_list=None):

        data_to_replace_with = self.init_data_to_replace_with(data)
        data = self.preprocess_data(data)        

        all_score_list = []
        all_sal_score_list = []

        for i in range(len(data)):
            
            if class_to_explain_list is None:
                class_to_explain = torch.argmax(model(data[i:i+1]),axis=1)[0]
            else:
                class_to_explain = class_to_explain_list[i]
        
            expl = explanations[i:i+1]
            left_pixel_nb = self.total_pixel_nb

            output = model(data[i:i+1])[0,class_to_explain]

            score_list = [output.detach().item()]
            saliency_score_list = []            
            iter_nb = 0

            while left_pixel_nb > 0:

                mask,saliency_scores = self.compute_mask(expl,data.shape,self.pixel_removed_per_step*(iter_nb+1))
                mask = mask.to(data.device)
                data_masked = self.apply_mask(data[i:i+1],data_to_replace_with[i:i+1],mask)

                output = model(data_masked)[0,class_to_explain]
                score_list.append(output.detach().item())
                saliency_score_list.append(saliency_scores[-self.pixel_removed_per_step:].detach().mean())
                    
                iter_nb += 1
                left_pixel_nb -= self.pixel_removed_per_step

            all_score_list.append(score_list)
            all_sal_score_list.append(saliency_score_list)

        all_score_list = np.array(all_score_list)
        all_sal_score_list = np.array(all_sal_score_list)

        mean_auc_metric = compute_auc_metric(all_score_list)
        mean_calibration_metric = self.compute_calibration_metric(all_score_list, all_sal_score_list)

        return self.make_result_dic(mean_auc_metric,mean_calibration_metric)

class DAUC(AUC_Metric):
    def __init__(self,data_shape,explanation_shape,data_transf_str="black",bound_max_step=True):
        super().__init__(data_shape,explanation_shape,data_transf_str,bound_max_step)
    
    def init_data_to_replace_with(self,data):
        return self.data_transf_func(data)

    def preprocess_data(self,data):
        return data

    def compute_calibration_metric(self, all_score_list, all_sal_score_list):
        score_var = all_score_list[:,:-1] - all_score_list[:,1:] 
        return compute_correlation(score_var, all_sal_score_list)

    def make_result_dic(self,auc_metric,calibration_metric):
        return {"dauc":auc_metric,"dc":calibration_metric}
        
class IAUC(AUC_Metric):
    def __init__(self,data_shape,explanation_shape,data_transf_str="blur",bound_max_step=True):
        super().__init__(data_shape,explanation_shape,data_transf_str,bound_max_step)
    
    def init_data_to_replace_with(self,data):
        return data

    def preprocess_data(self,data):
        return self.data_transf_func(data)

    def compute_calibration_metric(self, all_score_list, all_sal_score_list):
        score_var = all_score_list[:,1:] - all_score_list[:,:-1]
        return compute_correlation(score_var, all_sal_score_list)

    def make_result_dic(self,auc_metric,calibration_metric):
        return {"iauc":auc_metric,"ic":calibration_metric}