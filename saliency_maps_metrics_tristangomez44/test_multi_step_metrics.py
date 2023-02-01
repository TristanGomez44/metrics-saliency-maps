import torch
import numpy as np
from multi_step_metrics import DAUC,IAUC

#Test a model producing a linearly decreasing score from 1 to 0
def get_dauc1():
    test_dic = {}
    test_dic["data"] = torch.ones(1,3,56,56)
    test_dic["expl"] = test_dic["data"].clone()[:,0:1]
    test_dic["metrConst"] = DAUC
    def test_model_dauc_1(x):
        res_neg = (x==0).sum(dim=3).sum(dim=2).sum(dim=1,keepdim=True)
        res_pos = (x==1).sum(dim=3).sum(dim=2).sum(dim=1,keepdim=True)
        res = torch.cat((res_neg,res_pos),dim=1)
        res = res/res.sum(dim=1,keepdim=True)
        return res
    test_dic["model"] = test_model_dauc_1
    test_dic["class_to_explain"] = torch.ones(len(test_dic["data"])).long()
    test_dic["target"] = 0.5000
    test_dic["metric_name"] = "dauc"
    return test_dic 

#Test a model producing a approximately linearly increasing score from 0 to 1
def get_iauc1():
    test_dic = get_dauc1().copy()
    test_dic["metrConst"] = IAUC
    torch.manual_seed(0)
    data_shape = test_dic["data"].shape
    data = torch.rand(size=(data_shape[0],3,data_shape[2],data_shape[3]))
    test_dic["data"] = data
    test_dic["expl"] = test_dic["data"].clone()[:,0:1]
    data_ref_iauc1 = data.clone()
    def test_model_iauc_1(x):
        res_neg = (x!=data_ref_iauc1).sum(dim=3).sum(dim=2).sum(dim=1,keepdim=True)
        res_pos = (x==data_ref_iauc1).sum(dim=3).sum(dim=2).sum(dim=1,keepdim=True)
        res = torch.cat((res_neg,res_pos),dim=1)
        res = res/res.sum(dim=1,keepdim=True)
        return res
    test_dic["model"] = test_model_iauc_1
    test_dic["target"] = 0.5000
    test_dic["metric_name"] = "iauc"
    return test_dic

#Test a model producing a constant score
def get_dauc2():
    test_dic = {}
    data = torch.ones(1,3,56,56)
    test_dic["data"] = data.clone()
    test_dic["expl"]  = test_dic["data"].clone()[:,0:1]
    test_dic["metrConst"] = DAUC
    torch.manual_seed(2)
    rand_nb = torch.rand(size=(1,)).item()
    test_dic["model"]  = lambda x:rand_nb*torch.ones(x.shape[0],2)
    test_dic["class_to_explain"]  = torch.ones(len(test_dic["data"])).long()
    test_dic["target"] = rand_nb
    test_dic["metric_name"] = "dauc"
    return test_dic

#Test a model producing a constant score
def get_iauc2():
    test_dic = get_dauc2()
    test_dic = test_dic.copy()
    test_dic["metrConst"] = IAUC
    test_dic["metric_name"] = "iauc"
    return test_dic

#Test a fully uncalibrated model
def get_dc1():
    test_dic = {}
    data = torch.ones(1,3,56,56)
    test_dic["data"] = data.clone()
    data_shape = test_dic["data"].shape
    torch.manual_seed(2)
    expl = torch.rand(size=(data_shape[0],1,data_shape[2],data_shape[3]))
    test_dic["expl"] = expl.clone()
    test_dic["metrConst"] = DAUC
    def test_model_corr_metrics_1(x):
        res = torch.rand(size=(data.shape[0],2))
        return res
    test_dic["model"] = test_model_corr_metrics_1  
    test_dic["class_to_explain"] = torch.ones(data_shape[0]).long()
    test_dic["target"] = 0.00
    test_dic["metric_name"] = "dc"
    return test_dic 

#Test a fully uncalibrated model
def get_ic1():
    test_dic = get_dc1()    
    test_dic = test_dic.copy()
    test_dic["metrConst"] = IAUC
    test_dic["metric_name"] = "ic"
    return test_dic

#Generates random scores and a correlated explanation
def generate_random_scores_and_explanation(data_shape,metric):
    #Constructing random score variation
    torch.manual_seed(0)
    score_var,_ = torch.rand(size=(data_shape[2]*data_shape[3],)).sort(descending=True)
    score_var = score_var/score_var.sum()

    #Constructing associated scores
    scores = torch.zeros(score_var.shape[0]+1)
    scores[0] = score_var.sum() if metric == "DC" else 0
    
    for i in range(1,len(scores)):
        if metric == "DC":
            scores[i] = scores[i-1] - score_var[i-1]
        else:
            scores[i] = score_var[i-1] + scores[i-1]

    #Constructing correlated explanation
    torch.manual_seed(0)
    a,b = torch.rand(2)
    expl = a*score_var+b
    expl = expl.view(1,1,data_shape[2],data_shape[3])

    return scores,expl 

#Test a fully calibrated model
def get_dc2():
        
    test_dic = {}
    data = torch.ones(1,3,4,4)
    scores,expl = generate_random_scores_and_explanation(data.shape,"DC")

    metrConst = DAUC

    def test_model_dc_2(x):
        x = x[:,0:1].view(-1)
        inds = torch.argwhere(x==0)
        res = torch.zeros(1,2).float() 
        if len(inds) == 0:
            res[0,0] = scores[0]
            return res
        else:
            res[0,0] = scores[inds[-1,0]+1]
        return res

    model = test_model_dc_2  
    class_to_explain = torch.zeros(len(data)).long()            

    test_dic["data"] = data.clone()
    test_dic["expl"] = expl.clone()
    test_dic["metrConst"] = DAUC
    test_dic["model"] = model 
    test_dic["class_to_explain"] = class_to_explain
    test_dic["target"] = 0.9999
    test_dic["metric_name"] = "dc"
    return test_dic

def get_ic2():
    test_dic = get_dc2()    
    test_dic = test_dic.copy()
    test_dic["metrConst"] = IAUC
    scores_ic,expl = generate_random_scores_and_explanation(test_dic["data"].shape,"IC")
    test_dic["expl"] = expl
    data_ref = test_dic["data"].clone()
    def test_model_ic_2(x):
        x = x[:,0:1].view(-1)
        data_flat = data_ref[:,0:1].view(-1)
        inds = torch.argwhere(x==data_flat)
        res = torch.zeros(1,2).float() 
        if len(inds) == 0:
            res[0,0] = scores_ic[0]
            return res
        else:
            res[0,0] = scores_ic[inds[-1,0]+1]
        return res
    test_dic["model"] = test_model_ic_2
    test_dic["target"] = 0.9999
    test_dic["metric_name"] = "ic"
    return test_dic

if __name__ == "__main__":
 
    all_test_dic = {}
    all_test_dic["DAUC1"] = get_dauc1()
    all_test_dic["IAUC1"] = get_iauc1()
    all_test_dic["DAUC2"] = get_dauc2()
    all_test_dic["IAUC2"] = get_iauc2()
    
    all_test_dic["DC1"] = get_dc1()
    all_test_dic["IC1"] = get_ic1()
    all_test_dic["DC2"] = get_dc2()
    all_test_dic["IC2"] = get_ic2()
 
    test_to_do = ["DAUC1","IAUC1","DAUC2","IAUC2","DC1","IC1","DC2","IC2"]
    
    for test in test_to_do:
        torch.manual_seed(0)
        dic = all_test_dic[test]
        metric = dic["metrConst"](dic["data"].shape,dic["expl"].shape)
        mean = metric(dic["model"],dic["data"].clone(),dic["expl"].clone(),dic["class_to_explain"])[dic["metric_name"]]
        sucess = np.abs(mean - dic["target"]) < 0.01
        print(f"Test: {test}, Result:{mean}, Target:{dic['target']}, Sucess:{sucess}")