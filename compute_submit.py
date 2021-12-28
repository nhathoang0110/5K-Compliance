import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import shutil
import os

thres_mask=0.5
thres_dis=0.44

image_id=[]
fname=[]
for path in os.listdir("data"):
    image_id.append(int(path.split(".")[0]))
    fname.append(path)

d={'image_id':image_id,'fname':fname}
submit=pd.DataFrame(d)


mask1=pd.read_csv("output_csv/mask_swin_fold1_e12.csv")
mask2=pd.read_csv("output_csv/mask_swin_fold2_e13.csv")
mask3=pd.read_csv("output_csv/mask_swin_fold3_e7.csv")
mask4=pd.read_csv("output_csv/mask_swin_fold4_e9.csv")
mask5=pd.read_csv("output_csv/mask_swin_fold5_e14.csv")

distance1=pd.read_csv("output_csv/distancing_swin_fold1_e12.csv")
distance2=pd.read_csv("output_csv/distancing_swin_fold2_e11.csv")
distance3=pd.read_csv("output_csv/distancing_swin_fold3_e9.csv")
distance4=pd.read_csv("output_csv/distancing_swin_fold4_e10.csv")
distance5=pd.read_csv("output_csv/distancing_swin_fold5_e10.csv")



for path in os.listdir("data"):
    submit.loc[submit['fname']==path, 'mask1']=mask1.loc[mask1['fname']==path, 'mask'].item()
    submit.loc[submit['fname']==path, 'mask2']=mask2.loc[mask2['fname']==path, 'mask'].item()
    submit.loc[submit['fname']==path, 'mask3']=mask3.loc[mask3['fname']==path, 'mask'].item()
    submit.loc[submit['fname']==path, 'mask4']=mask4.loc[mask4['fname']==path, 'mask'].item()
    submit.loc[submit['fname']==path, 'mask5']=mask5.loc[mask5['fname']==path, 'mask'].item()

list_submit=[]
list_submit.append(submit['mask1'])
list_submit.append(submit['mask2'])
list_submit.append(submit['mask3'])
list_submit.append(submit['mask4'])
list_submit.append(submit['mask5'])
submit['mask']=[0]*len(os.listdir("data"))
for i in list_submit:
    submit['mask']+=i
submit['mask']/=len(list_submit)


for path in os.listdir("data"):
    submit.loc[submit['fname']==path, 'distancing1']=distance1.loc[distance1['fname']==path, 'distancing'].item()
    submit.loc[submit['fname']==path, 'distancing2']=distance2.loc[distance2['fname']==path, 'distancing'].item()
    submit.loc[submit['fname']==path, 'distancing3']=distance3.loc[distance3['fname']==path, 'distancing'].item()
    submit.loc[submit['fname']==path, 'distancing4']=distance4.loc[distance4['fname']==path, 'distancing'].item()
    submit.loc[submit['fname']==path, 'distancing5']=distance5.loc[distance5['fname']==path, 'distancing'].item()

list_submit=[]
list_submit.append(submit['distancing1'])
list_submit.append(submit['distancing2'])
list_submit.append(submit['distancing3'])
list_submit.append(submit['distancing4'])
list_submit.append(submit['distancing5'])
submit['distancing']=[0]*len(os.listdir("data"))
for i in list_submit:
    submit['distancing']+=i
submit['distancing']/=len(list_submit)


submit['5K']= ((submit['mask']>thres_mask) & (submit['distancing']>thres_dis)).astype(int)
submit_final = submit[['image_id', 'fname', '5K']]


submit_final=submit_final.sort_values(by=['image_id'])
submit_final.to_csv('result/submission.csv',index=False)








