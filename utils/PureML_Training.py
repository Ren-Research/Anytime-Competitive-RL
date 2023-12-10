import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np


import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from loss import efficiency_cost, QoS_cost, revenue_cost
from parameters import system_parameters

# System parameters 
#---------------------------------------------- 
H = system_parameters.H

power_convert = system_parameters.power_convert
D1 = system_parameters.D1
D2 = system_parameters.D2
D3 = system_parameters.D3
C1 = system_parameters.C1
C2 = system_parameters.C2
gamma = system_parameters.gamma

Lambda = system_parameters.Lambda
Budget= system_parameters.Budget

size_X = system_parameters.size_X
size_D = system_parameters.size_D

V_d = system_parameters.V_d
alpha = system_parameters.alpha

# Load and preprocess training data
#---------------------------------------------- 

def load_data():
    demand_org = sio.loadmat('../data/demand.mat')['train']
    renewables = sio.loadmat('../data/renewables.mat')['train']

    transit_noise_a = sio.loadmat('../data/transit_noise_1.mat')['train']
    transit_noise_x = sio.loadmat('../data/transit_noise_2.mat')['train']

    H = system_parameters.H
    batch_size = 50

    context_train = np.concatenate([np.reshape(demand_org,[-1,1,H]),np.reshape(renewables,[-1,1,H]),
                                   np.reshape(transit_noise_a,[-1,1,H]),np.reshape(transit_noise_x,[-1,1,H])],axis=1)
    train_dataloader = DataLoader(context_train, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_dataloader


# Policy network definition
#---------------------------------------------- 
class Net(torch.nn.Module):
    def __init__(self,D_in,D_out,k1,k2):
        super(Net, self).__init__()
        self.input_linear = nn.Linear(D_in, k1)
        self.hidden_linear = nn.Linear(k1, k2)
        self.output_linear = nn.Linear(k2, D_out)
        torch.nn.init.kaiming_normal_(self.input_linear.weight,nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.hidden_linear.weight,nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.output_linear.weight,nonlinearity='relu')
        

    def base_forward(self, base_in_c,base_in_b, base_in_s):
        base_in=torch.cat([base_in_c, base_in_b, base_in_s], -1)
        h1_relu = F.relu(self.input_linear(base_in))
        h2_relu = F.relu(self.hidden_linear(h1_relu))
        y_pred = F.relu(self.output_linear(h2_relu))
        return y_pred
    
    def policy_forward(self, policy_in_c, trans_noise, demand_noise, action_pre, state_pre):
        action_ = action_pre
        state_ = state_pre
        action_out_list = []
        seq_len = policy_in_c.size(1)
        for t in range(seq_len):
            policy_t_c = policy_in_c[:,t]
            noise_t = trans_noise[:,t]
            noise2_t = demand_noise[:,t]
            action_out = self.base_forward(policy_t_c,action_,state_)
            action_out_list.append(action_out)
            action_ = action_out
            state_ = torch.minimum(F.relu(state_ * (1-noise2_t) + policy_t_c[:,0].unsqueeze(1) - (power_convert+noise_t)*action_out),
                                  torch.tensor(15))
        
        action_out=torch.cat(action_out_list,-1)
        return action_out
    
# training
#---------------------------------------------- 
def ML_training(net, train_dataloader, num_epoch, learn_rate):
    loss_scale = 0.01
    train_loss_list=[]
    
    for epoch in range(num_epoch):
        if epoch==50:
            optimizer.param_groups[0]['lr']=0.5*learn_rate
        elif epoch==80:
            optimizer.param_groups[0]['lr']=0.1*learn_rate
        for _, context in enumerate(train_dataloader):
            optimizer.zero_grad()
            seq_len = context.size(2)
            policy_in_c = torch.transpose(context[:,:2], 1,2)
            trans_noise = context[:,2].view(-1, seq_len, 1)
            demand_noise = context[:,3].view(-1, seq_len, 1)
            action_pre = torch.zeros([policy_in_c.size(0),1])
            state_pre = torch.zeros([policy_in_c.size(0),1])
            action_out=net.policy_forward(policy_in_c,trans_noise,demand_noise,action_pre,state_pre)
            action_out = action_out.unsqueeze(2)

            # evaluate loss
            efficiency_loss = efficiency_cost(policy_in_c[:,:,1].unsqueeze(2),action_out, C1, C2)
            rev_loss = revenue_cost (trans_noise, action_out ,V_d, alpha)
            train_loss = (efficiency_loss + rev_loss * gamma)*loss_scale
            train_loss_list.append(train_loss.detach().numpy())
            train_loss.backward()
            optimizer.step()

        print('epoch: %d,training loss: %.4f' %(epoch,train_loss) )
    return net


if __name__ == "__main__": 
    train_dataloader = load_data()
                       
    torch_seed=1
    torch.random.manual_seed(torch_seed)
                       
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # Initialize the policy net
    global_hiddens = [40,40]
    net=Net(4,1,global_hiddens[0],global_hiddens[1]).double().to(device)
    
                       
    # Learning parameters
    num_epoch = 100
    learn_rate=1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)
    
    # training
    model = ML_training(net, train_dataloader, num_epoch, learn_rate)  
    
    # save model
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    PATH = '../runs_new/'+'rnn_'+current_time
    torch.save(model.state_dict(), PATH)