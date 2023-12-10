import torch
import torch.nn.functional as F


def efficiency_cost(renewable,action, C1,C2):
    batch_size = renewable.size(0)
    hit_cost = F.relu(action-renewable)*C1
    # hit_cost = torch.sum(hit_cost)
    # hit_cost = (action-renewable)*C1
    hit_cost = torch.sum(hit_cost**2)

    #switch_diff = action[:,1:,:] - action[:,:-1,:]
    switch_cost = torch.norm(action[:,1:,:] - action[:,:-1,:], dim=2)
    switch_cost = torch.sum(switch_cost**2)
    # switch_cost = torch.sum(switch_cost)

    # if c < 1e4:
    #     total_loss = scale*(hit_cost + c * switch_cost)
    # else:
    #     total_loss = scale*switch_cost
    
    total_loss = hit_cost + C2 * switch_cost

    total_loss /= batch_size

    return total_loss



def QoS_cost(demand,trans_noise, demand_noise, action, D1,D2,D3, size_X=15):
    batch_size = demand.size(0)
    seq_len = demand.size(1)
    x_pre = torch.zeros([batch_size,1])
    total_loss = 0
    state_list=[]
    
    for h in range(seq_len):
        x = torch.minimum(F.relu(x_pre * (1-demand_noise[:,h]) + demand[:,h] - (0.8+trans_noise[:,h])*action[:,h]),torch.tensor(size_X))
        total_loss += torch.sum (D1*x**2+D2*x+D3)
        x_pre = x
        state_list.append(x_pre)

    total_loss /= batch_size

    return total_loss, torch.stack(state_list, dim=1)

def revenue_cost (trans_noise, action, V_d, alpha=0.5):
    batch_size = trans_noise.size(0)
    return  -torch.sum((V_d**alpha) * ((0.8 + trans_noise) * action)**(1-alpha))/ batch_size
