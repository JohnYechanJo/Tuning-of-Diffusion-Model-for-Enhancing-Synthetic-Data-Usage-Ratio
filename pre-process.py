import os
import csv
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import gc
from time import *
import re
from PIL import Image
from torchvision import transforms
from transformers import ViTModel
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# the num for a demo of each category/ binary classification
# 64:64 as a batch/ maybe 10 batches in a demo eg.
# might need to keep the balance in a train batch and mkae the order random
# please ensure the num is suitable in advance or change the code to add some caculating part
# default: maybe there should be a config 
pic_num = 640
batch_size = 128
batch_num = 10
half_batch = 64

def data_process():
    print("data_process: Start!")
    # my starting position is in the upper directory of the database
    # codes below can be flexibly changed
    train_path_CNV = os.getcwd()+"/archive/OCT2017/train/CNV"
    train_path_NORM = os.getcwd()+"/archive/OCT2017/train/NORMAL"
    # basic trans into tensor list
    CNV_tensor_list = load_trans(train_path_CNV)
    NORM_tensor_list = load_trans(train_path_NORM)
    # 4 outputs in [pic_num,768]
    out_00,out_01,out_02,out_03 = VIT_process(CNV_tensor_list)
    out_10,out_11,out_12,out_13 = VIT_process(NORM_tensor_list)

    # compose
    tensor_list_0 = []
    tensor_list_1 = []
    tensor_list_2 = []
    tensor_list_3 = []
    labels_list = []
    for i in range(batch_num):
        cnv_tensor_0 = out_00[i*half_batch:(i+1)*half_batch]
        norm_tensor_0 = out_10[i*half_batch:(i+1)*half_batch]
        cnv_tensor_1 = out_01[i*half_batch:(i+1)*half_batch]
        norm_tensor_1 = out_11[i*half_batch:(i+1)*half_batch]
        cnv_tensor_2 = out_02[i*half_batch:(i+1)*half_batch]
        norm_tensor_2 = out_12[i*half_batch:(i+1)*half_batch]
        cnv_tensor_3 = out_03[i*half_batch:(i+1)*half_batch]
        norm_tensor_3 = out_13[i*half_batch:(i+1)*half_batch]
        # [batch_size,768]
        tensor_0 = torch.cat((cnv_tensor_0,norm_tensor_0),dim=0)
        tensor_1 = torch.cat((cnv_tensor_1,norm_tensor_1),dim=0)
        tensor_2 = torch.cat((cnv_tensor_2,norm_tensor_2),dim=0)
        tensor_3 = torch.cat((cnv_tensor_3,norm_tensor_3),dim=0)
        # [batch_size]
        labels = torch.cat([torch.zeros(half_batch, dtype=torch.long), torch.ones(half_batch, dtype=torch.long)], dim=0)
        # shuffle
        indices = torch.randperm(batch_size)
        tensor_list_0.append(tensor_0[indices])
        tensor_list_1.append(tensor_1[indices])
        tensor_list_2.append(tensor_2[indices])
        tensor_list_3.append(tensor_3[indices])
        labels_list.append(labels[indices])
    tensor_set_0 = torch.cat(tensor_list_0,dim=0)
    tensor_set_1 = torch.cat(tensor_list_1,dim=0)
    tensor_set_2 = torch.cat(tensor_list_2,dim=0)
    tensor_set_3 = torch.cat(tensor_list_3,dim=0)
    labels_set = torch.cat(labels_list,dim=0)

    # save
    if os.path.exists('pre-trained_dataset.pt'):
        os.remove('pre-trained_dataset.pt')
    torch.save({'data_0':tensor_set_0,'data_1':tensor_set_1,'data_2':tensor_set_2,'data_3':tensor_set_3,'label':labels_set},'pre-trained_dataset.pt')
    print('data_process: Done \
           Please Check')



    
def load_trans(path):
    trans_toTensor = img_transform()
    image_list = []
    i=0
    # traverse the image
    for filename in os.listdir(path):
        if i==pic_num:
            break
        file_path = os.path.join(path, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(file_path).convert("RGB")
                tensor_img = trans_toTensor(img)
                image_list.append(tensor_img)
            except Exception as e:
                print(f"Skip: {filename}, Error: {e}")
        i+=1
    # [N, C, H, W] if tensor
    # VIT: 224*224
    return image_list


def img_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.crop((0, 100, 768, 400))),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def VIT_process(img_list):
    # 16 inputs per time prevent memory explosion from the hardware limitations 
    model_name = "google/vit-base-patch16-224"
    VIT_model = ViTModel.from_pretrained(model_name, output_hidden_states=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VIT_model.to(device)
    VIT_model.eval()

    length = len(img_list)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []

    for i in range(int(length/16)):
        batch_imgs = img_list[i*16:(i+1)*16]
        batch_tensor = torch.stack(batch_imgs,dim=0).to(device)
        with torch.no_grad():
            outputs = VIT_model(pixel_values=batch_tensor)
        hidden = outputs.hidden_states
        # optain the outputs of hidden layers
        # init to the same shape as the last layer CLS
        # [size,768]
        last_layer = hidden[12][:,0,:]
        hidden_layer_1 = torch.zeros_like(last_layer)
        hidden_layer_2 = torch.zeros_like(last_layer)
        hidden_layer_3 = torch.zeros_like(last_layer)

        for j in range(12):
            # divide 12 hidden layers into 4 group / add up each and take the average
            # proved in the paper https://dl.acm.org/doi/abs/10.1145/3404835.3462871 and my last program though for Bert
            # but I think it may useful in Transformer based models, lets try
            # 0-3 layers CLS
            if j<4:
                hidden_layer_1+=hidden[j][:,0,:]/4
            # 4-7 layers CLS
            elif j<8:
                hidden_layer_2+=hidden[j][:,0,:]/4
            # 8-11 layers CLS
            else:
                hidden_layer_3+=hidden[j][:,0,:]/4
        
        layers_0.append(last_layer)
        layers_1.append(hidden_layer_1)
        layers_2.append(hidden_layer_2)
        layers_3.append(hidden_layer_3)
        gc.collect()
        torch.cuda.empty_cache()
    layers_0 = torch.cat(layers_0,dim=0)
    layers_1 = torch.cat(layers_1,dim=0)
    layers_2 = torch.cat(layers_2,dim=0)
    layers_3 = torch.cat(layers_3,dim=0)
    return layers_0,layers_1,layers_2,layers_3

data_process()