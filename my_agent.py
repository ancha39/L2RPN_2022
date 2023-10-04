import torch
from grid2op.Agent import BaseAgent
from grid2op.Reward import BaseReward
import sys
import os, copy
import numpy as np
from copy import deepcopy
from .agent_file import PPO



redispatch_acts_arr  =  []
battery_acts_arr     =  []

class MyAgent(BaseAgent):  
    def __init__(self, action_space, all_actions, this_directory_path='./'):
        BaseAgent.__init__(self, action_space=action_space)
        
        self.actions                 = np.load(os.path.join(this_directory_path, 'ActionSpace/action_id_l2rpn_wcci.pickle'), allow_pickle = True)
        self.obs_chosen              = list(range(1, 3718))        
        self.normalization_values    = np.load(os.path.join(this_directory_path, 'ActionSpace/normalization_values_.pickle'), allow_pickle = True)
        self.redispatch_acts         = np.load(os.path.join(this_directory_path, 'ActionSpace/redispatch_vect_full.pickle'), allow_pickle = True)
        self.battery_actions         =  np.load(os.path.join(this_directory_path, 'ActionSpace/battery_actions.pickle'), allow_pickle = True)
        device                       = torch.device('cpu')
        in_dim                       = 3717
        out_dim                      = len(np.load(os.path.join(this_directory_path, 'ActionSpace/action_id_l2rpn_wcci.pickle'), allow_pickle=True))
        self.agent                   = PPO(in_dim, out_dim,device, coef_entropy=1e-3, coef_value_func=0.01)
        self.reconnet_flag           = False
        self.all_actions             = all_actions 

    def load(self, dir_path='./'):
        self.agent.load(dir_path)
        
    def convert_array2act(self, observation, total_array):
        return total_array
    
    def legal_check(self, action, obs):
        act_dict = action.as_dict()                                
        if 'change_bus_vect' not in act_dict:                         
            return True
        substation_to_operate = int(act_dict['change_bus_vect']['modif_subs_id'][0])
        if obs.time_before_cooldown_sub[substation_to_operate]:   
            return False
        for line in [eval(key) for key, val in act_dict['change_bus_vect'][str(substation_to_operate)].items()
                     if 'line' in val['type']]:
            if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                return False
        return True

    
    def reward_function(self,obs, action, has_error, is_done, is_illegal, is_ambiguous,is_redispatch):
        
        self.reward_min = -2
        if is_done or is_illegal or is_ambiguous or has_error or is_redispatch:
            return self.reward_min
        
        offline_line_cnt = len(np.where(obs.line_status != 1)[0]) - len(np.where(obs.duration_next_maintenance >0)[0])
        if np.max(obs.rho) >= 1:
            idc_ = np.where(obs.rho>=1)[0]
            rho_ = np.sum(obs.rho[idc_]- 0.5)
        else:
            rho_ = np.max(np.max(obs.rho)-0.5,0)
        reward = np.exp((-rho_- 0.5*offline_line_cnt))
        return reward
    


    def line_sub_impacted(self, action):      
        topology_action_on_lines = []
        topology_action_on_subid = None 
        for i in range(len(action.impact_on_objects()['topology']['bus_switch'])):           
            object_type = action.impact_on_objects()['topology']['bus_switch'][i]['object_type']

            if object_type == 'line (extremity)' or object_type == 'line (origin)':
                object_id = action.impact_on_objects()['topology']['bus_switch'][i]['object_id']
                topology_action_on_lines.append(object_id)

        return topology_action_on_lines    
    

    def act(self, observation, reward, done):

        line_rho_threshold = 0.97
        
        if observation.rho.max() <= line_rho_threshold:              
            a =  self.action_space({})
            return self.check_reconnect_act(observation, a,-2)

        action                                 = self.action_space({})
        sim_obs_, sim_rew, sim_done_, sim_info = observation.simulate(self.action_space({}))
        
        is_done         = sim_done_
        is_illegal      = sim_info['is_illegal']
        is_ambiguous    = sim_info['is_ambiguous']
        is_redispatch   = sim_info['is_dispatching_illegal']
        has_error       = len(sim_info['exception'])
        
        baseline_reward = self.reward_function(sim_obs_,action, has_error, is_done, is_illegal, is_ambiguous,is_redispatch)
        baseline_rho    = sim_obs_.rho.max()
        act_selected    = self.action_space({})
        idx_selected    = None
        
        feat            = observation.to_vect()[self.obs_chosen]
        features        = self.normalize_ip(feat)
        _, pred_act, _  = self.agent.model.Model(features)                                  

        pred_act        = pred_act.detach().numpy()
        redispatch_flag = False
        battery_flag    = False
        sorted_actions  = np.flip(np.argsort(pred_act)) 
        
        for k, idx in enumerate(sorted_actions):
            act_enc = self.convert_array2act(observation, self.all_actions[int(self.actions[idx])])
            
            sim_obs, sim_rew, sim_done, sim_info = observation.simulate(act_enc) 
            is_done                              = sim_done
            is_illegal                           = sim_info['is_illegal']
            is_ambiguous                         = sim_info['is_ambiguous']
            is_redispatch                        = sim_info['is_dispatching_illegal']
            has_error                            = len(sim_info['exception'])
            sim_reward                           = self.reward_function( sim_obs, action, has_error, is_done, is_illegal, is_ambiguous,is_redispatch)
             
            if sim_done or is_illegal or is_ambiguous or is_redispatch:
                continue                
                
            if sim_reward > baseline_reward: #sim_obs.rho.max() < min_rho:                                       
                baseline_reward = sim_reward    
                act_selected    = copy.deepcopy(act_enc)
                idx_selected    = idx
                baseline_rho    = sim_obs.rho.max()

        # topology + redispatch action
        topology_act = copy.deepcopy(act_selected)
        
        if baseline_rho > 0.98:
            idxb_selected = None 
            for idxb, i in enumerate(self.battery_actions):
                
                battery_act = self.action_space.from_vect(i)
                final_act = battery_act + topology_act
                sim_obs, sim_rew, sim_done, sim_info = observation.simulate(final_act) 
                  
                is_done       = sim_done
                is_illegal    = sim_info['is_illegal']
                is_ambiguous  = sim_info['is_ambiguous']
                is_redispatch = sim_info['is_dispatching_illegal']
                has_error     = len(sim_info['exception'])
                sim_reward    = self.reward_function(sim_obs, action, has_error, is_done, is_illegal, is_ambiguous,is_redispatch)      
                
                if sim_done or is_illegal or is_ambiguous or is_redispatch:
                    continue                

                if sim_reward > baseline_reward:                                       
                    baseline_reward = sim_reward 
                    act_selected    = copy.deepcopy(final_act)
                    battery_flag    = True
                    battery_act     = self.action_space({})
                    idxb_selected   = copy.deepcopy(idxb)
                    baseline_rho    = sim_obs.rho.max()

            if idxb_selected is not None:
                battery_acts_arr.append(idxb_selected)         
        topology_act = copy.deepcopy(act_selected)
        
        if baseline_rho > 0.98:
            idxg_selected = None
            for idx,i in enumerate(self.redispatch_acts):
                redispatch_act = self.action_space.from_vect(i)
                
                final_act = redispatch_act + topology_act

                sim_obs, sim_rew, sim_done, sim_info = observation.simulate(final_act) 
                is_done       = sim_done
                is_illegal    = sim_info['is_illegal']
                is_ambiguous  = sim_info['is_ambiguous']
                is_redispatch = sim_info['is_dispatching_illegal']
                has_error     = len(sim_info['exception'])
                sim_reward    = self.reward_function(sim_obs, action, has_error, is_done, is_illegal, is_ambiguous,is_redispatch)                

                if sim_done or is_illegal or is_ambiguous or is_redispatch:
                    continue                

                if sim_reward > baseline_reward:                                       
                    baseline_reward = sim_reward
                    act_selected    = copy.deepcopy(final_act)
                    redispatch_flag = True
                    redispatch_act  = self.action_space({})
                    idxg_selected   = copy.deepcopy(idx)
                    baseline_rho    = sim_obs.rho.max()

            if idxg_selected is not None:
                redispatch_acts_arr.append(idxg_selected)                   
            
        return self.check_reconnect_act(observation, act_selected,baseline_reward) if act_selected else self.check_reconnect_act(observation, self.action_space({}),baseline_reward)

 

    def check_reconnect_act(self, obs, original_action,baseline_reward):
        disconnect_flag = False
        disc_flag       = False
        reco_flag       = False
        disconnect_topo_act = original_action
        reconnect_topology_action = original_action
        tested_actions = copy.deepcopy(original_action)

        disc_act        = copy.deepcopy(original_action)
        fall_back_act   = copy.deepcopy(original_action)
        disconnected_lines = np.where(obs.line_status == False)[0]    
        
        
        rho = copy.deepcopy(obs.rho)
        overflow = copy.deepcopy(obs.timestep_overflow)
        to_disc = (rho >=1.0) & (overflow == 3)
        to_disc[rho >2.0] = True   
        sim_reward_ld     = -5
        
        if not len(disconnected_lines) and not len(to_disc):        
            print("debugging here")
            return original_action
        
        if (obs.time_before_cooldown_line[disconnected_lines] > 0).all() and not len(to_disc):    
            print("stuck at cooldown")
            return original_action
        
        line_to_reconnect = -1
        line_to_disconnect = -1
                    
        for line in disconnected_lines:            
            if not obs.time_before_cooldown_line[line]:
                reconnect_array       = np.zeros_like(obs.rho).astype(int)
                reconnect_array[line] = 1
                reconnect_action      = deepcopy(original_action)
                vect_original_act     = original_action.to_vect()
                reconnect_action      = self.act_update(obs, vect_original_act,reconnect_array)
                
                sim_obs, sim_rew,  sim_done, sim_info  = obs.simulate(reconnect_action)
                
                is_done               = sim_done
                is_illegal            = sim_info['is_illegal']
                is_ambiguous          = sim_info['is_ambiguous']
                is_redispatch         = sim_info['is_dispatching_illegal']
                has_error             = len(sim_info['exception'])
                sim_reward_rc         = self.reward_function(sim_obs, reconnect_action, has_error, is_done, is_illegal, is_ambiguous,is_redispatch)
                    
                if sim_reward_rc     >= baseline_reward:
                    line_to_reconnect = line
                    baseline_reward   = copy.deepcopy(sim_reward_rc)
                    reco_flag         = True
                    
        if line_to_reconnect != -1:
            reconnect_array = np.zeros_like(sim_obs.rho).astype(int)
            reconnect_array[line_to_reconnect] = 1
            original_action = self.act_update(obs, vect_original_act,reconnect_array)    
            reconnect_topology_action = original_action    


        
        if np.any(to_disc):
            for id_ in np.where(to_disc)[0]:
                
                disconnect_array       = np.zeros_like(obs.rho).astype(int)
                disconnect_array[id_] = -1
                # disconnect_action      = deepcopy(tested_actions)
                vect_original_act      = tested_actions.to_vect()
                disconnect_flag        = True
                disconnect_action      = self.action_space({'set_line_status': disconnect_array})
                tested_act         = disconnect_action + tested_actions
                
                sim_obs, sim_rew,  sim_done, sim_info  = obs.simulate(tested_act)
                is_done               = sim_done
                is_illegal            = sim_info['is_illegal']
                is_ambiguous          = sim_info['is_ambiguous']
                is_redispatch         = sim_info['is_dispatching_illegal']
                has_error             = len(sim_info['exception'])

                if not sim_done and not is_illegal and not is_ambiguous and not is_redispatch:                    
                    sim_reward_ld         = self.reward_function(sim_obs, tested_act, has_error, is_done, is_illegal, is_ambiguous,is_redispatch)
                    
                    if sim_reward_ld      >= baseline_reward:
                        print("disconnection_actions")
                        disc_flag         = True
                        disc_act          = copy.deepcopy(tested_act)   
                        line_to_disconnect= id_

                        
            if line_to_disconnect != -1 and disc_flag:
                disconnect_topo_act =  copy.deepcopy(disc_act)   
                
                
        if reco_flag:
            return reconnect_topology_action
        elif disc_flag:
            return disconnect_topo_act
        else:
            return fall_back_act
            
    def act_update(self, obs, total_array, reconnect_array):
        disconnected_lines      = np.where(obs.line_status == False)[0]  
        disc_line_cooldown_zero = []
        
        for i in disconnected_lines:
            if obs.time_before_cooldown_line[i] == 0:
                disc_line_cooldown_zero.append(i)
        
        reconnect_line = np.where(reconnect_array > 0)[0]                             
        action         = self.action_space.from_vect(total_array)
        act_lines      = self.line_sub_impacted(action)                                                   

        if np.any(act_lines == reconnect_line): 
            print("breaking here")
            print("action", action)
            return action       
        else: 
            line_reconnection =  self.action_space({'set_line_status': reconnect_array})
            final_action = line_reconnection + action
            sim_obs, sim_rew, sim_done, sim_info = obs.simulate(final_action) 
                  
            is_done       = sim_done
            is_illegal    = sim_info['is_illegal']
            is_ambiguous  = sim_info['is_ambiguous']
            is_redispatch = sim_info['is_dispatching_illegal']
            has_error     = len(sim_info['exception'])
            
            sim_reward    = self.reward_function(sim_obs, final_action, has_error, is_done, is_illegal, is_ambiguous,is_redispatch)      
                
            if not sim_done and not is_illegal and not is_ambiguous and not is_redispatch:
                return final_action
            
        return action

    def normalize_ip(self, obss):
        obss[:1]  = obss[:1]/12     # month/12
        obss[1:2] = obss[1:2]/31    # date/31
        obss[2:3] = obss[2:3]/24    # hour/24
        obss[3:4] = obss[3:4]/60    # minute/60
        obss[4:5] = obss[4:5]/7     # day of week

        obss[5:67]      = obss[5:67]  /self.normalization_values['gen_p']           # generator p
        obss[67:129]    = obss[67:129] /  self.normalization_values['gen_q']        # generator q
        obss[129:191]   = obss[129:191] /  self.normalization_values['gen_v']       # generator voltage 

        obss[191:282]   = obss[191:282] /  self.normalization_values['load_p']         # load p
        obss[282:373]   =  obss[282:373] /  self.normalization_values['load_q']        # load q
        obss[373:464]   = obss[373:464] /  self.normalization_values['load_v']         # load v

        obss[464:650]   = obss[464:650] / self.normalization_values['line_p_or']        # origin line p
        obss[650:836]   = obss[650:836]  / self.normalization_values['line_q_or']       # origin line load
        obss[836:1022]  = obss[836:1022] / self.normalization_values['line_v_or']       # origin voltage
        obss[1022:1208] = obss[1022:1208] / self.normalization_values['line_a_or']      # origin voltage
        obss[1208:1394] = obss[1208:1394]  / self.normalization_values['line_p_ex']     # p ex
        obss[1394:1580] = obss[1394:1580]  / self.normalization_values['line_q_ex']     # q ex
        obss[1580:1766] = obss[1580:1766]  / self.normalization_values['line_v_ex']     # v_ex
        obss[1766:1952] = obss[1766:1952]  / self.normalization_values['line_a_ex']     # a_extremity
        obss[1952:2138] = obss[1952:2138]         # line rho
        obss[2138:2324] = obss[2138:2324]         # line status 
        obss[2324:2510] = obss[2324:2510]/ 25     # timestep overflow
        obss[2510:3042] = obss[2510:3042]         # overall topology 
        obss[3042:3228] = obss[3042:3228]/ 25     # line cooldown
        obss[3228:3346] = obss[3228:3346]/ 25     # substation cooldown
        obss[3346:3532] = obss[3346:3532]/ 11000  # maintanance
        obss[3532:3718] = obss[3532:3718]/ 100    # duration next maintenance 
        return obss    

    
def make_agent(env, this_directory_path):
    all_actions = env.action_space.get_all_unitary_topologies_change(env.action_space)
    my_agent = MyAgent(env.action_space, all_actions ,this_directory_path)
    my_agent.load(this_directory_path)
    return my_agent