import sys
import numpy as np
import torch
import torch.nn.functional as F

def compute_corr_metric(score_var, all_sal_score_list):
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

def blur_data(data):
    kernel_size = max(data.shape[2]//5,9)
    kernel = torch.ones(kernel_size,kernel_size)
    kernel = kernel/kernel.numel()
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,1,-1,-1)
    kernel = kernel.to(data.device)
    data = F.conv2d(data,kernel,padding=kernel.size(-1)//2,groups=kernel.size(0))  
    return data

class AUC_Metric():

    def __init__(self,data_shape,explanation_shape,bound_max_step=True):

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

    def update_data(self,data, i, y1, y2, x1, x2,data_to_replace_with):
        data[i:i+1,:,y1:y2,x1:x2] = data_to_replace_with[i:i+1,:,y1:y2,x1:x2]
        return data

    def data_to_replace_with(data):
        raise NotImplementedError

    def preprocess_data(data):
        raise NotImplementedError

    def compute_metric(self,all_score_list,all_sal_score_list):
        return compute_auc_metric(all_score_list)

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
                data_masked = data[i:i+1]*mask + data_to_replace_with[i:i+1]*(1-mask)

                output = model(data_masked)[0,class_to_explain]
                score_list.append(output.detach().item())
                saliency_score_list.append(saliency_scores[-self.pixel_removed_per_step:].detach().mean())
                    
                iter_nb += 1
                left_pixel_nb -= self.pixel_removed_per_step

            all_score_list.append(score_list)
            all_sal_score_list.append(saliency_score_list)

        all_score_list = np.array(all_score_list)
        all_sal_score_list = np.array(all_sal_score_list)

        mean = self.compute_metric(all_score_list,all_sal_score_list)
        return mean

    def compute_mask(self,explanation,data_shape,k):
        mask_flat = torch.ones(data_shape[2]*data_shape[3])
        saliency_scores,inds = torch.topk(explanation.view(-1),k,0,sorted=True)
        mask_flat[inds] = 0
        mask = mask_flat.reshape(1,1,data_shape[2],data_shape[3])

        mask = torch.nn.functional.interpolate(mask,data_shape[2:],mode="nearest")[0]
        return mask,saliency_scores

class DAUC(AUC_Metric):
    def __init__(self,data_shape,explanation_shape,bound_max_step):
        super().__init__(data_shape,explanation_shape,bound_max_step)
    
    def init_data_to_replace_with(self,data):
        return torch.zeros_like(data,device=data.device)

    def preprocess_data(self,data):
        return data

class IAUC(AUC_Metric):
    def __init__(self,data_shape,explanation_shape,bound_max_step):
        super().__init__(data_shape,explanation_shape,bound_max_step)
    
    def init_data_to_replace_with(self,data):
        return data

    def preprocess_data(self,data):
        return blur_data(data)

class DC(DAUC):
    def __init__(self,data_shape,explanation_shape,bound_max_step):
        super().__init__(data_shape,explanation_shape,bound_max_step)
    
    def compute_metric(self, all_score_list, all_sal_score_list):
        score_var = all_score_list[:,:-1] - all_score_list[:,1:] 
        return compute_corr_metric(score_var, all_sal_score_list)

class IC(IAUC):
    def __init__(self,data_shape,explanation_shape,bound_max_step):
        super().__init__(data_shape,explanation_shape,bound_max_step)
    
    def compute_metric(self, all_score_list, all_sal_score_list):
        score_var = all_score_list[:,1:] - all_score_list[:,:-1]
        return compute_corr_metric(score_var, all_sal_score_list)