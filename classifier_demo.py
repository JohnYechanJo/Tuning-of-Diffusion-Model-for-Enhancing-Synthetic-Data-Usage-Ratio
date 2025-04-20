import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.init as init
import argparse
import pickle
import json
import json, os, time
import argparse
import random
import gc
from time import *
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# we can load all the data and divide it into 8:1:1 as train/val/test three sets in sequence
# 'data_0''data_1''data_2''data_3' & 'label'

# Only when the classifier itself has sufficient performance can we better evaluate its impact, 
# so we need to find a better classifier structure, although classifier complexity is closely related to data complexity

file_path = os.path.join('pre-trained_dataset.pt')
if os.path.exists(file_path):
    dataset_dic = torch.load(file_path)
    print("Load successfully!")
else:
    print("File not exist!")
# maybe a config file later
total = 1280
train = 1024
val = 128
test = 128
batch_size = 128
epochs = 15
dropout_rate = 0.6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = dataset_dic['label']

# deal with self-attention
class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm = False,attn_dropout=0):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size
        self.is_layer_norm = is_layer_norm
        if self.is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))
        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)
        V_att = Q_K_score.bmm(V)
        return V_att
    
    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()
        #[batchsize,len,inputsize] * [inputsize,n_heads*dim_k] 
        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, k_len, self.n_heads, self.d_v)
        
        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, k_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, k_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        #[bsz,q_len,inputsize]
        return output

    def forward(self, Q, K, V):
        V_att = self.multi_head_attention(Q, K, V)
        
        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X#残差
        return output

# Add embedding data into gradient flow and do basic trans-size using residual MLP
class EncoderBlock(nn.Module):
    def __init__(self, input_dim = 768, output_dim = 300,hidden_dim_1 = 300,hidden_dim_2 = 450,attn_drop =0.15):
        super(EncoderBlock,self).__init__()
        self.attn_drop = attn_drop
        #[size,768] shuffled pre batch
        embedding_weights_0 = dataset_dic['data_0']
        embedding_weights_1 = dataset_dic['data_1']
        embedding_weights_2 = dataset_dic['data_2']
        embedding_weights_3 = dataset_dic['data_3']
        self.embedding_layer_0 = nn.Embedding(num_embeddings = total,embedding_dim=input_dim
                                            ,padding_idx= 0 
                                            ,_weight = embedding_weights_0)
        self.embedding_layer_1 = nn.Embedding(num_embeddings = total,embedding_dim=input_dim
                                            ,padding_idx= 0 
                                            ,_weight = embedding_weights_1)
        self.embedding_layer_2 = nn.Embedding(num_embeddings = total,embedding_dim=input_dim
                                            ,padding_idx= 0 
                                            ,_weight = embedding_weights_2)
        self.embedding_layer_3 = nn.Embedding(num_embeddings = total,embedding_dim=input_dim
                                            ,padding_idx= 0 
                                            ,_weight = embedding_weights_3)
        self.linear_1 = nn.Linear(input_dim,hidden_dim_1)
        self.linear_2 = nn.Linear(hidden_dim_1,hidden_dim_2)
        self.linear_3 = nn.Linear(hidden_dim_2,output_dim)
        self.dropout = nn.Dropout(attn_drop)
        
        self.relu = nn.ReLU()

        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.linear_1.weight)
        init.xavier_normal_(self.linear_2.weight)
        init.xavier_normal_(self.linear_3.weight)
    
    #generate idex in sequence and change it to tensor to vist
    def forward(self, layer_id=0, X_id=0):
        if torch.is_tensor(X_id):
            X_id.to(device)
            if(layer_id==0):
                X_ = self.embedding_layer_0(X_id).to(torch.float32)
            elif(layer_id==1):
                X_ = self.embedding_layer_1(X_id).to(torch.float32)
            elif(layer_id==2):
                X_ = self.embedding_layer_2(X_id).to(torch.float32)
            elif(layer_id==3):
                X_ = self.embedding_layer_3(X_id).to(torch.float32)
        else:
            print("Non-standard use of encoderblock!")
        
        #residual connect in MLP
        residual = self.relu(self.linear_1(X_))
        x_ = self.relu(self.dropout(self.linear_2(residual)))
        x_ = self.linear_3(x_)+residual
        
        return x_
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def forward(self):
        pass

    def fit(self,x_train,y_train,x_val,y_val,x_test,y_test):
        if torch.cuda.is_available():
            self.cuda()
        # set learning rate
        self.optimizer = torch.optim.Adam(self.parameters(),lr=8e-5,weight_decay=0)
        dataset = TensorDataset(x_train,y_train)
        dataloader = DataLoader(dataset,batch_size,shuffle=False)
        loss = nn.CrossEntropyLoss()
        # training epochs
        for epoch in range(epochs):
            print("\nEpoch ", epoch + 1, "/", epochs)
            self.train()
            # Convert to a batched iterator
            for i ,data in enumerate(dataloader):
                total = len(dataloader)
                # unbind
                batch_x_id,batch_y = (item.cuda(device = self.device)for item in data)
                self.batch_dealer(batch_x_id,batch_y,loss,i,epoch+1,total)
            # validation part
            self.batch_evaluate(x_val,y_val)
    
    # deal with  x & y batches in training part
    def batch_dealer(self,x_id,y,loss,i,epoch,total):
        # clean previous grad
        self.optimizer.zero_grad()
        logit_original = self.forward(x_id,epoch=epoch)
        loss_classify = loss(logit_original,y)
        loss_classify.backward()
        self.optimizer.step()
        corrects = (torch.max(logit_original, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total
                                                                            ,loss_classify.item()
                                                                            ,accuracy
                                                                            ,corrects
                                                                            ,y.size(0)))
    
    def batch_evaluate(self,x,y):
        y_pred = self.predicter(x)
        acc = accuracy_score(y,y_pred)
        if acc>self.best_acc:
            self.best_acc = acc
        print(classification_report(y, y_pred, target_names=['NR', 'FR'], digits=5))
        print("Val set acc:", acc)
        print("Best val set acc:", self.best_acc)
        
        
    def predicter(self,x):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset,batch_size=16)
        for i,data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_id = data[0].cuda(device = self.device)
                logits = self.forward(batch_x_id)
                predicted = torch.max(logits,dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred
        
    

class Classifier(NeuralNetwork):
    def __init__(self):
        super(Classifier,self).__init__()
        self.encoder_block = EncoderBlock()
        self.attention = TransformerBlock(input_size=300)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        # classifier struct
        self.fc = nn.Linear(1200,300)
        self.fc1 = nn.Linear(300,600)
        self.fc2 = nn.Linear(600,300)
        self.fc3 = nn.Linear(in_features=300, out_features=2)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.fc.weight)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)

    # every batch
    def forward(self,x_id,epoch=0):
        batch_size = x_id.shape[0]
        x_id.cuda()
        # [bsz,300]
        embedding_0=self.encoder_block(layer_id =0,X_id=x_id)
        embedding_1=self.encoder_block(layer_id =1,X_id=x_id)
        embedding_2=self.encoder_block(layer_id =2,X_id=x_id)
        embedding_3=self.encoder_block(layer_id =3,X_id=x_id)
        # combining information from various hidden layers
        # [bsz,1200] -> [bsz,300]
        embedding = self.relu(self.fc(torch.cat( (embedding_0
                                                 ,embedding_1
                                                 ,embedding_2
                                                 ,embedding_3),dim=1)))
        # trans to a form suitable to the transformer block [bsz,1,300]
        enhanced = self.attention(embedding.view(batch_size, 1, 300),embedding.view(batch_size, 1, 300)
                                     ,embedding.view(batch_size, 1, 300))
        # [bsz,300]
        enhanced = enhanced.squeeze(1)
        a1 = self.relu(self.dropout(self.fc1(enhanced)))
        a1 = self.relu(self.dropout(self.fc2(a1)))
        output = self.fc3(a1)
        return output
        


def train_and_test(model):
    # use this as a substitute(index)
    x_train = torch.arange(0,1024)
    x_val = torch.arange(1024,1152)
    x_test = torch.arange(1152,1280)
    y_train = labels[0:1024]
    y_val = labels[1024:1152]
    y_test = labels[1152:1280]
    nn=model
    #train and val
    nn.fit(x_train,y_train,x_val,y_val,x_test,y_test)
    #test part
    y_pred = nn.predicter(x_test)
    res = classification_report(y_test, y_pred, target_names=['NR', 'FR'], digits=3, output_dict=True)
    for k, v in res.items():
        print(k, v)
    print("result:{:.4f}".format(res['accuracy']))
    print(res)
    # end
    return res

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
model = Classifier()
train_and_test(model)
