import torch
import numpy as np
from single_step_metrics import IIC_AD,ADD

#Test a model producing a score that increases when the image is masked
def get_iic1():
    test_dic = {}
    test_dic["data"] = torch.ones(10,3,56,56)
    expl = torch.zeros_like(test_dic["data"])[:,0:1]
    expl[0,0,0,0] = 1
    test_dic["expl"] = expl
    test_dic["metrConst"] = IIC_AD
    def test_model_iic1(x):
        res = torch.zeros(1,2)
        res[0,0] = (x==0).float().mean()
        return res
    test_dic["model"] = test_model_iic1
    test_dic["class_to_explain"] = torch.zeros(len(test_dic["data"])).long()
    test_dic["target"] = 1
    test_dic["metric_name"] = "iic"
    return test_dic 

#Test a model producing a score that decreases when the image is masked
def get_iic2():
    test_dic = get_iic1()
    def test_model_iic2(x):
        res = torch.zeros(1,2)
        res[0,0] = (x==1).float().mean()
        return res
    test_dic["model"] = test_model_iic2
    test_dic["target"] = 0
    return test_dic

#Test a model producing a random score
def get_iic3():
    test_dic = get_iic1()
    def test_model_iic3(x):
        res = torch.rand(size=(x.shape[0],2))
        return res
    test_dic["model"] = test_model_iic3
    test_dic["target"] = 0.5
    return test_dic  

#Test a model with a score that increases when the explanation is applied
def get_ad1():
    test_dic = {}
    test_dic["data"] = torch.ones(10,3,56,56)
    expl = torch.zeros_like(test_dic["data"])[:,0:1]
    torch.manual_seed(0)
    baseline = torch.rand(size=(1,)).item()
    test_dic["baseline"] = baseline
    test_dic["expl"] = expl
    test_dic["metrConst"] = IIC_AD
    def test_ad1(x):
        res = torch.zeros(1,2)
        res[0,0] = baseline+(x==0).float().mean(dim=(1,2,3))
        res = torch.softmax(res,dim=-1) 
        return res
    test_dic["model"] = test_ad1
    test_dic["class_to_explain"] = torch.zeros(len(test_dic["data"])).long()
    test_dic["target"] = 0
    test_dic["metric_name"] = "ad"
    return test_dic 

#Test a model with a score that decreases of a constant proportion when the explanation is applied
def get_ad2():
    test_dic = get_ad1()
    baseline = test_dic["baseline"]
    torch.manual_seed(1)
    drop_prop = torch.rand(size=(1,)).item()
    test_dic["drop_prop"] = drop_prop
    def test_ad2(x):
        res = torch.zeros(1,2)
        if (x==0).sum() == 0:
            res[0,0] = baseline
        else:
            res[0,0] = baseline*drop_prop
        return res
    test_dic["model"] = test_ad2
    test_dic["target"] = 1-drop_prop
    return test_dic

#Test a model with a constant score
def get_add1():
    test_dic = get_ad1()
    def test_add1(x):
        res = torch.ones(1,2)
        return res 
    test_dic["model"] = test_add1
    test_dic["metrConst"] = ADD
    test_dic["metric_name"] = "add"
    return test_dic 

#Test a model with a score that decreases of a constant proportion when the explanation is applied
def get_add2():
    test_dic = get_ad2()
    expl = test_dic["expl"]
    expl[:,:,0,0] = 1
    test_dic["metrConst"] = ADD
    return test_dic 

if __name__ == "__main__":
 
    all_test_dic = {}
    all_test_dic["IIC1"] = get_iic1()
    all_test_dic["IIC2"] = get_iic2()
    all_test_dic["IIC3"] = get_iic3()
    all_test_dic["AD1"] = get_ad1()
    all_test_dic["AD2"] = get_ad2()
    all_test_dic["ADD1"] = get_add1()
    all_test_dic["ADD2"] = get_add2()
    test_to_do = ["IIC1","IIC2","IIC3","AD1","AD2","ADD1","ADD2"]
    
    for test in test_to_do:
        torch.manual_seed(0)
        dic = all_test_dic[test]
        metric = dic["metrConst"]()
        mean = metric(dic["model"],dic["data"].clone(),dic["expl"].clone(),dic["class_to_explain"])
        sucess = np.abs(mean - dic["target"]) < 0.01
        print(f"Test: {test}, Result:{mean}, Target:{dic['target']}, Sucess:{sucess}")
