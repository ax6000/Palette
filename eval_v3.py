import argparse
from core.base_dataset import BaseDataset
from models.metric import inception_score
import glob
import numpy as np
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate
import json
def calc_min_max(x):
    # x = (x.astype(np.float32)/127.5-1)
    return np.nanmin(x,axis=1),np.nanmax(x,axis=1)

# def create_plots(x,y):
#     n=4
#     x *= 200
#     y *= 200
#     fig,axes = plt.subplots(2,n,figsize=(10,5),tight_layout=True)
#     for i in range(n):
#         axes[0,i].plot(x[i,:])
#         axes[1,i].plot(y[i,:])
#     plt.show()
#     plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -s ..\..\data\processed\BP_npy\<dir name>\p00\scale_test.npy 
    # -d .\experiments\<dir name>\results\test\<epoch no>
    parser.add_argument('-s', '--src', type=str, help='experiment directory')
    parser.add_argument('-e', '--epoch', type=str, help='epoch',default='0')
   
    ''' parser configs '''
    args = parser.parse_args()
    n_patients = 0
    errors = []
    # open json and select scale file
    with open(os.path.join(args.src,"config.json"), "r") as file:
        json_data = json.load(file)
    
    scale_dir = json_data["datasets"]["train"]["which_dataset"]["args"]["data_root"]
    print(scale_dir)
    
    # load scale file
    scales = np.load(os.path.join(scale_dir,"scale_train.npy"))
    result_dir = os.path.join(args.src,"results","test",args.epoch)
    gt = []
    cond= []
    out = []
    gt_files = glob.glob(f"{result_dir}\GT_*.npy")
    cond_files = glob.glob(f"{result_dir}\Process_*.npy")
    out_files = glob.glob(f"{result_dir}\OUT_*.npy")
    for j in tqdm(range(len(gt_files))):
        gt.append(np.load(os.path.join(gt_files[j])))
        cond.append(np.load(os.path.join(cond_files[j])))
        out.append(np.load(os.path.join(out_files[j])))
    print(gt[-1].shape)
    out = np.concatenate(out,axis=0)
    gt = np.concatenate(gt,axis=0)
    cond = np.concatenate(cond,axis=0)
    scales = scales[:out.shape[0]]
    print(scales.shape,gt.shape,out.shape,scales)
    # 正規化を元に戻す
    gt[:] -= scales[0,0]
    gt[:] /= scales[0,1]
    out[:] -= scales[0,0]
    out[:] /= scales[0,1]
    gt_mean = np.mean(gt.flatten())
    out_mean = np.mean(out.flatten())
    cond_mean = np.mean(cond.flatten())
    gt_std = np.std(gt.mean(axis=1),dtype=np.float64)
    out_std = np.std(out.mean(axis=1),dtype=np.float64)
    cond_std = np.std(cond.mean(axis=1),dtype=np.float64)
    print(np.count_nonzero(np.isnan(out)))
    
    headers = ["Signal", "Mean","Std"]
    table=[]
    table.append(["cond_ppg",cond_mean,cond_std])
    # table.append(["data_ppg", 0.494153162946643,0.10694360087091538])
    # table.append(["data_abp",0.39593121533751857,0.13489903083932583])
    table.append(["gt",gt_mean,gt_std])
    table.append(["out",out_mean,out_std])
    print(tabulate(table,headers, floatfmt=".4f"))
    # 血圧の最大値(SBP)最小値(DBP)を計算
    gt_min,gt_max = calc_min_max(gt)
    out_min,out_max = calc_min_max(out)
    # 誤差
    errors = np.zeros((2,*out_min.shape))
    errors[0,:]=gt_min-out_min
    errors[1,:]=gt_max-out_max
    errors = errors[:,~np.isnan(errors).any(axis=0)]
    print(np.count_nonzero(np.isnan(errors)))
    print("error shape:",errors.shape)
    n_samples = errors.shape[1]
    print("n_samples:",n_samples)
    me = np.mean(errors,axis=1)
    mae = np.mean(np.abs(errors),axis=1)
    rmse = np.sqrt(np.mean(errors**2,axis=1))
    std = np.std(errors,axis=1)
    print(me.shape,mae.shape,rmse.shape,std.shape)
    error_5 = np.count_nonzero(np.abs(errors)<=5,axis=1)/n_samples*100
    error_15 = np.count_nonzero(np.abs(errors)<=15,axis=1)/n_samples*100
    error_10 = np.count_nonzero(np.abs(errors)<=10,axis=1)/n_samples*100
    print("""
          test data samples:
          # samples : {}
          
          Eval Stats:   SBP    DBP
          MAE:        {:6.3f} {:6.3f}
          RMSE:       {:6.3f} {:6.3f}
          Mean Error: {:6.3f} {:6.3f}
          STD:        {:6.3f} {:6.3f}
          
          BHS standards range:
          Error   <5mmHg <10mmHg <15mmHg
          gradeA     60%     85%     95%
          gradeB     50%     75%     90%
          gradeC     40%     65%     85%
          SBP     {:5.1f}%  {:5.1f}%  {:5.1f}%
          DBP     {:5.1f}%  {:5.1f}%  {:5.1f}%
           
          
          """.format(
            n_samples,
            *mae,
            *rmse,
            *me,
            *std,
            error_5[0], error_10[0], error_15[0],
            error_5[1], error_10[1], error_15[1],
          ))
    # print(n_samples,np.count_nonzero(np.abs(errors)<=5,axis=1),np.count_nonzero(errors<=10,axis=1),np.count_nonzero(errors<=15,axis=1))