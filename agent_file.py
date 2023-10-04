import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import pickle
import sys


class PPO(nn.Module): 
    def __init__(self,in_dim, out_dim,device ,coef_entropy=0.01, coef_value_func=0.5, max_grad_norm=0.5):
        super(PPO, self).__init__()
        
        device =torch.device('cpu')
        self.device =  torch.device('cpu')
        self.model = Policy_Value_Network(in_dim,out_dim,device)
        self.optimizer =  optim.Adam(self.model.Model.parameters()) 
        self.coef_entropy = coef_entropy
        self.coef_value_func = coef_value_func
        self.max_grad_norm = max_grad_norm
        self.step = self.model.step
        self.value = self.model.value
        self.initial_state = None
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approx_kl', 'clip_ratio']


    def train(self, obs, returns, actions, values, neg_log_p_old, advs, lr = 3e-4, clip_range = 0.2):
        kll = {}
        obs = obs

        l, p, _ = self.model.Model(obs)
        
        actions_one_hot = F.one_hot(actions.long(), num_classes=list(l.shape)[-1])     

        # calculating the log probability of the logits 
        neg_log_p = F.cross_entropy(l.to("cpu"), actions.long().to("cpu")) 

        # calculate entropy bonus
        entropy = torch.mean(self._get_entropy(l)) 

        # calculate value loss
        vpred = self.model.value(obs) #model value 
        vpred_clip = values.to("cpu") +torch.clip(vpred.to("cpu") - values.to("cpu"), -clip_range, clip_range)        # clipping the value function 

        vpred = vpred
        returns = returns
        vpred_clip = vpred_clip
        value_loss1 = torch.square(vpred - returns)     
        value_loss2 = torch.square(vpred_clip - returns)   

        value_loss = 0.5 * torch.mean(torch.maximum(value_loss1, value_loss2))

        # calculate policy loss
        neg_log_p_old = neg_log_p_old
        neg_log_p     = neg_log_p
        ratio = torch.exp(neg_log_p_old - neg_log_p)                                    # ratio of the old policy to the new policy obtained during the transitions  

        policy_loss1 = -advs * ratio.to("cpu")                                                    # the ratio of policy is multiplied by the advantage of the episode
        policy_loss2 = -advs * torch.clip(ratio.to("cpu"), (1 - clip_range), (1 + clip_range))    # clippiing the policy update, so that remove incentives if the new policy is too far from the old policy
        policy_loss = torch.mean(torch.max(policy_loss1, policy_loss2))

        approx_kl = 0.5 * torch.mean(torch.square(neg_log_p_old - neg_log_p))
        clip_ratio = torch.mean((torch.greater(torch.abs(ratio.to("cpu") - 1), clip_range)).type(torch.float))

        # Sigma loss
        loss = policy_loss * 10 + value_loss * self.coef_value_func - entropy * self.coef_entropy
        # print("sigma loss", loss)

        # loss
        self.optimizer.zero_grad()                                                      # model optimizer, Adam optimizer is used
        loss.backward()                                                                 # calculating the gradients

        nn.utils.clip_grad_value_(self.model.Model.parameters(), clip_value=1.0)

        self.optimizer.step()                                                           # losses are backpropogated for weight updates
        
        return policy_loss, value_loss, entropy, approx_kl, clip_ratio

                  
    def save(self, dir_path, epoch):                                                    # function for save the model during training
        torch.save(self.model.Model.state_dict(), os.path.join(dir_path, 'epoch_ckpt-{}.pt'.format(epoch)),map_location=torch.device('cpu'))

   
    def load(self, dir_path):                                                           # function for loading the model during inference
        self.model.Model.load_state_dict(torch.load(os.path.join(dir_path, 'saved_ckpt/epoch_ckpt.pt'),map_location=torch.device('cpu')))

    def _get_entropy(self,l):                                                           # function used to calculate the entropy bonus of the logits 
        a0 = l - torch.max(l)
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0)
        p0 = ea0/z0
        return torch.sum(p0* (torch.log(z0) - a0))


class Policy_Value_Network(object):
    
    def __init__(self, in_dim, out_dim,device): 
        self.device = torch.device("cpu")
        self.Model = PVNet(in_dim, out_dim,device)
        self.device = device

    def step(self, obs):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("lllkkk",(obs.shape))

        obs = obs
        l, p, v = self.Model(obs)                                                       # from the observation, actor model gives logits, probability distribution, while critic model gives us the value
        
        u  = torch.rand(l.shape)                                                    # the Gumbel-max trick used to select the action from the probability distribution
        a = torch.argmax(l - torch.log(-torch.log(u)), axis = -1) 
        a_one_hot = (F.one_hot(a,num_classes=list(l.shape)[-1])).type('torch.FloatTensor')


        target_indices = torch.argmax(a_one_hot, dim=1)
        neg_log_p = F.cross_entropy(l, target_indices.long())

        neg_log_p = neg_log_p.unsqueeze(0)
        v = torch.squeeze(v, axis = 1)

        return a, v, neg_log_p, l                                                       # during the high rho value, model gives us act_id, value, probability distribution, and logits                                                       
    
    def value(self, obs):
        _, _, v = self.Model(obs)                                                       # critic function used to evaluates how good actions were taken by actor model
        v = torch.squeeze(v, axis=1)                                            
        return v

class PVNet(nn.Module):
    def __init__(self, in_dim, out_dim,device):
        super(PVNet, self).__init__()
        self.device = device
        hidden_dim = 2000

        self.layer1     = nn.Linear(in_dim, hidden_dim)
        self.layer2     = nn.Linear(hidden_dim,hidden_dim)
        self.layer3     = nn.Linear(hidden_dim,hidden_dim)
        self.layer4     = nn.Linear(hidden_dim,hidden_dim)
        self.layer5     = nn.Linear(hidden_dim,hidden_dim)
        self.act_layer  = nn.Linear(hidden_dim, out_dim)

        # single layer critic network, with hidden dimension of [64,1], where one is the dimension of the output
        self.val_hidden_layer    = nn.Linear(hidden_dim, 64)
        self.val_layer  = nn.Linear(64,1)


    def forward(self, obs):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs = obs

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # ReLU activation function is used to produce the decision of actor model, and critic model
        s = F.relu(self.layer1(obs))
        s = F.relu(self.layer2(s))
        s = F.relu(self.layer3(s))
        s = F.relu(self.layer4(s))
        s = F.relu(self.layer5(s))
        l = self.act_layer(s)                           # logits
        
        
        #print("max_logits", torch.max(l))
                
        p = F.softmax(l, dim = 0)                       # probability distribution of actions
        # print("max probability distribution of actions",torch.max(p))
        vh = F.relu(self.val_hidden_layer(s))           # (1000-> 64) (nn.Relu)
        v = self.val_layer(vh)                          # state value(64-> 1)
        
        return l, p, v
