import torch
import torch.nn.functional as F

def opt_QoS (state_h, demand_h, power_convert):
    return (state_h+demand_h)/power_convert

def opt_QoS_seq (demand,trans_noise):
    batch_size = demand.size(0)
    seq_len = demand.size(1)
    x_pre = torch.zeros([batch_size,1])
    total_loss = 0 
    state_list = []
    action_list = []

    for h in range(H):
        action = opt_QoS(x_pre, demand[:,h])
        action_list.append(action)
        x = F.relu(x_pre + demand[:,h] - (power_convert+trans_noise[:,h])* action)
        total_loss += torch.sum (D1*x**2+D2*x+D3)
        x_pre = x
        state_list.append(x)
        
    total_loss /= batch_size 
    state_list = torch.stack(state_list, dim=1)
    action_list = torch.stack(action_list, dim=1)

    return state_list, action_list, total_loss