from ruamel.yaml import YAML, dump, RoundTripDumper
from raisim_gym.env.RaisimGymVecEnv import RaisimGymVecEnv as Environment
from raisim_gym.env.env.hummingbird import __HUMMINGBIRD_RESOURCE_DIRECTORY__ as __RSCDIR__
from raisim_gym.algo.ppo2 import PPO2
from raisim_gym.archi.policies import MlpPolicy
from raisim_gym.helper.raisim_gym_helper import ConfigurationSaver, TensorboardLauncher
from _raisim_gym import RaisimGymEnv
import os
import math
import argparse
import random, math
import numpy as np
from prop_fault_detection.fault_detection import fault_detection
import matplotlib
# matplotlib.use('GTK3Agg') 
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
detect_ste = -1
prop_val = 0

font = {'size'   : 30}

matplotlib.rc('font', **font)

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default=os.path.abspath(__RSCDIR__ + "/default_cfg.yaml"),
                    help='configuration file')
parser.add_argument('--scene', type=int, default=0,
                    help='Enter and integer from 0 to 4 where 0 -> 4 propeller system; 1 -> 3 propeller system;"+\
                        " 2 -> 3 propeller system; 3 -> 4 to 3 propeller system; 4 -> 3 to 2 propeller system. Defaults to 0.')
parser.add_argument('--plot', type=int,default=0,
                    help='0 to supress plotting and 1 to enable plotting. Defaults to 0.')
args = parser.parse_args()
mode = args.scene
cfg_abs_path = parser.parse_args().cfg
cfg = YAML().load(open(cfg_abs_path, 'r'))

fd = fault_detection()

# create environment from the configuration file
cfg['environment']['num_envs'] = 1
env_no = cfg['environment']['num_envs']
env = Environment(RaisimGymEnv(__RSCDIR__, dump(cfg['environment'], Dumper=RoundTripDumper)))


base_path = "/home/rohit/Documents/raisim_stuff/prop_loss_final/quadcopter_weights/"
weight_prop = [base_path + "4_working_prop/2020-06-01-07-50-00_Iteration_4500.pkl",\
               base_path + "3_working_prop/2020-05-31-13-36-06_Iteration_4500.pkl",\
               base_path + "2_working_prop/2020-05-30-07-37-15_Iteration_4500.pkl"]

model_list = []
for i in range(3):
	model_list.append(PPO2.load(weight_prop[i]))


obs = env.reset()
running_reward = 0.0
ep_len = 0

switch_off1 = 1000
switch_off2 = 1000

pos = np.zeros((15,3), dtype = np.float32)
pos_plot = np.zeros((2000,3), dtype = np.float32)
setpoint = np.zeros((2000,3), dtype = np.float32)
setpoint[1000:,1] = 1.0
# setpoint[:,-1] = -9.0
angV_plot = np.zeros((2000,3), dtype = np.float32)

model_pred = np.zeros((2000,1), dtype = np.float32)

final_mean = np.zeros((1,3), dtype = np.float32)
stab_count = 0
prev = False

count = 0
prev_p = 0

obs_append_43 = np.zeros((1,100,18), dtype=obs.dtype)
obs_append_43[:,-1,:] = obs
prop_loss_cum_43 = np.zeros((100,1), dtype = np.float32)

obs_append_32 = np.zeros((1,200,18), dtype=obs.dtype)
prop_loss_cum_32 = np.zeros((200,1), dtype = np.float32)

prop_loss_check = [False, False, False, False]
# prop_loss_check = [False, False, False, True]


def check_prop_loss(prop_loss_cum):
    uq = np.unique(prop_loss_cum[50:,:]).size
    if(uq <= 1):
        return int(prop_loss_cum[-1,0])
    else:
        return 0
    
def take_action_43(obs, ste):
    global detect_ste
    action = np.zeros((env_no,4), dtype = np.float32)
    res = [j for j, val in enumerate(prop_loss_check) if val]
    obs_append_43[:,:99,:] = obs_append_43[:,1:,:]
    obs_append_43[-1,:] = obs[:]
    if(ste>150 and len(res) == 0):
        prop_loss = main_43.predict(obs_append_43)
        prop_loss_cum_43[:99,:] = prop_loss_cum_43[1:,:]
        prop_loss_cum_43[-1,:] = np.argmax(prop_loss)
        prop_val = check_prop_loss(prop_loss_cum_43) # use model of prop loss detection

        # if(int(prop_val[i,:,:]) == 1):
        #     prop_loss_check[i][int(prop_val[i,:,:]-1)] = True
        # elif(int(prop_val[i,:,:]) == 2):
        #     prop_loss_check[i][int(prop_val[i,:,:]-1)] = True
        # elif(int(prop_val[i,:,:]) == 3):
        #     prop_loss_check[i][int(prop_val[i,:,:]-1)] = True
        # elif(int(prop_val[i,:,:]) == 4):
        #     if(detect_ste < 0):
        #         detect_ste = ste
        #     prop_loss_check[i][int(prop_val[i,:,:]-1)] = True
        
        if(prop_val == 1):
            prop_loss_check[prop_val-1] = True
        elif(prop_val == 2):
            prop_loss_check[prop_val-1] = True
        elif(prop_val == 3):
            prop_loss_check[prop_val-1] = True
        elif(prop_val == 4):
            if(detect_ste < 0):
                detect_ste = ste
            prop_loss_check[prop_val-1] = True

    if(ste<switch_off1):
        action, _ = model_list[0].predict(obs)
    else:
        action, _ = model_list[0].predict(obs)
        action[:,-1] = -1.1
        if(len(res) == 1):
            if(res == 0):
                action, _ = model_list[1].predict(obs)
                action = np.insert(action, 0, -1.1, axis = 1)
            elif(res == 1):
                action, _ = model_list[1].predict(obs)
                action = np.insert(action, 1, -1.1, axis = 1)
            elif(res == 2):
                action, _ = model_list[1].predict(obs)
                action = np.insert(action, 2, -1.1, axis = 1)
            elif(res == 3):
                action, _ = model_list[1].predict(obs)
                action = np.insert(action, 3, -1.1, axis = 1)            
    action = action * 2.5 + 1.5
    return action

def take_action_32(obs, ste):
    global detect_ste
    action = np.zeros((env_no,4), dtype = np.float32) - 1.25
    for i in range(env_no):
        res = [j for j, val in enumerate(prop_loss_check) if val]
        obs_append_32[i,:199,:] = obs_append_32[i,1:,:]
        obs_append_32[i,-1,:] = obs
        if(ste>250 and len(res) == 1):
            prop_loss = main_32.predict(obs_append_32)
            prop_loss_cum_32[:199,:] = prop_loss_cum_32[1:,:]
            prop_loss_cum_32[-1,:] = np.argmax(prop_loss)
            prop_val = check_prop_loss(prop_loss_cum_32)
            if(prop_val == 1):
                if(detect_ste < 0):
                    detect_ste = ste
                prop_loss_check[2] = True
        # print(prop_loss_check)
        if(ste<switch_off2):
            action, _ = model_list[1].predict(obs)
            action = np.insert(action, 3, -1.25, axis = 1)
        else:
            action, _ = model_list[1].predict(obs)
            action[:,2] = -1.25
            action = np.insert(action, 3, -1.25, axis = 1)
            if(len(res) == 2):
                if(res == [2,3]):
                    action, _ = model_list[2].predict(obs)
                    action = np.insert(action, 2, -1.25, axis = 1)
                    action = np.insert(action, 3, -1.25, axis = 1)
    # print(action)
    # action = action * 2.5 + 1.5
    return action


def take_action(mode, obs, ste):
    if(mode == 0):
        action, _ = model_list[0].predict(obs)
        return action
    elif(mode == 1):
        action, _ = model_list[1].predict(obs)
        action = np.insert(action, 3, -1.25, axis = 1)
        return action
    elif(mode == 2):
        action, _ = model_list[1].predict(obs)
        action = np.insert(action, 2, -1.25, axis = 1)
        action = np.insert(action, 3, -1.25, axis = 1)
        return action
    elif(mode == 3):
        return take_action_43(obs, ste)
    elif(mode == 4):
        return take_action_32(obs, ste)

env.start_recording_video("tp.mp4")
try:
    for ste in range(2000):
        env.wrapper.showWindow()

        assert 0 <= mode <= 4, "Mode should be 0, 1, 2, 3 or 4"
        action = take_action(mode, obs, ste)

        obs, reward, done, infos = env.step(action, visualize=True)
        # print(obs[:,11])
        
        angV_plot[ste,:] = obs[:,15:]
        # print(obs[:,9:12])
        pos_plot[ste,:] = obs[:,9:12] 
        pos[:14,:] = pos[1:,:]
        pos[-1,:] = obs[:,9:12]
        mean = np.mean(pos - setpoint[ste,:], axis=0)
        std = np.std(pos, axis=0)
        
        if(ste>150):
            # if((std<0.005).all()):
            #     if(prev == True):
            #         stab_count += 1
            #     prev = True
            # else:
            #     prev = False
            # if(stab_count > 10):
            #     final_mean = mean * 1.1
            
            obs[:,9:12] += mean 
        running_reward += reward[0]
        ep_len += 1
        if done:
            print("Episode Reward: {:.2f}".format(running_reward))
            print("Episode Length", ep_len)
            running_reward = 0.0
            ep_len = 0
        obs[:,9:12] -= setpoint[ste,:]
        if(ste == 3500-1):
            np.save("pos_plot.npy", pos_plot)
except KeyboardInterrupt:
    env.stop_recording_video()

# plot horizontal
fig, (ax1, ax2) = plt.subplots(2,1)
p1, = ax1.plot(np.arange(np.size(pos_plot[:,0]), dtype=np.int32)/100 , pos_plot[:,0], label='actual', linewidth=3, color='#0000ff')
p2, = ax1.plot(np.arange(np.size(pos_plot[:,0]),  dtype=np.int32)/100, setpoint[:,0], linestyle='--', dashes=(3, 3), label='setpoint', linewidth=4, color='#008000')
v1 = ax1.axvline(x=10.0, color='black', linestyle='-.', linewidth=3, label = 'Loss of single propeller')
v2 = ax1.axvline(x=detect_ste/100, color='r', linewidth=3, label = 'Detected loss of single propeller')
ax1.set_ylabel("Hori. X Pos. [m]")
# l1 = ax1.legend([p1,p2], ["actual","setpoint"], loc=2, frameon=False)
# ax1.add_artist(l1)
# l2 = ax1.legend([v1,v2], ["Single prop. lost", "Detection"], loc=4, frameon=False)

p1, = ax2.plot(np.arange(np.size(pos_plot[:,1]), dtype=np.int32)/100 , pos_plot[:,1], label='actual', linewidth=3, color='#0000ff')
p2, = ax2.plot(np.arange(np.size(pos_plot[:,0]),  dtype=np.int32)/100, setpoint[:,1], linestyle='--', dashes=(3, 3), label='setpoint', linewidth=4, color='#008000')
v1 = ax2.axvline(x=10.0, color='black', linestyle='-.', linewidth=3, label = 'Loss of single propeller')
v2 = ax2.axvline(x=detect_ste/100, color='r', linewidth=3, label = 'Detected loss of single propeller')
ax2.set_ylabel("Hori. Y Pos. [m]")
ax2.set_xlabel("Timestep [sec]")
l1 = ax2.legend([p1,p2], ["actual","setpoint"], loc=4, frameon=False)
ax2.add_artist(l1)
# l2 = ax2.legend([v1,v2], ["Single prop. lost", "Detection"], loc=3, frameon=False)
ymin, ymax = ax2.get_ylim() 
ylim2 = max(abs(ymin),abs(ymax)) 

ymin, ymax = ax1.get_ylim() 
ylim1 = max(abs(ymin),abs(ymax))

ylim = max(ylim1,ylim2)
ax1.set_ylim([-ylim,ylim])
ax2.set_ylim([-ylim,ylim])

# plt.legend()
# plt.tight_layout()
# figure = plt.gcf() # get current figure
# figure.set_size_inches(16, 12)
# plt.savefig("/home/rohit/Documents/raisim_stuff/images/2_xy.png", bbox_inches = 'tight',
#     pad_inches = 0, dpi=150)

# plot height z
fig, (ax2, ax1) = plt.subplots(2,1)
pos_plot[:,2] = (pos_plot[:,2]/2) + 5.0
# print(pos_plot[:,2])
p1, = ax1.plot(np.arange(np.size(pos_plot[:,2]), dtype=np.int32)/100 , (pos_plot[:,2]), label='actual', linewidth=3, color='#0000ff')
p2, = ax1.plot(np.arange(np.size(pos_plot[:,2]),  dtype=np.int32)/100, setpoint[:,2]/2 + 5, linestyle='--', dashes=(3, 3), label='setpoint', linewidth=4, color='#008000')
v1 = ax1.axvline(x=10.0, color='black', linestyle='-.', linewidth=3, label = 'Loss of single propeller')
v2 = ax1.axvline(x=detect_ste/100, color='r', linewidth=3, label = 'Detected loss of single propeller')
# ax1.yaxis.set_label_coords(-0.05,0.32)
ax1.set_ylabel("Height above\n Ground [m] ")
ax1.set_yticks(np.arange(6))
ax1.set_xlabel("Timestep [sec]")

l1 = ax1.legend([p1,p2], ["actual","setpoint"], frameon=False)
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])
# ax1.add_artist(l1)
# l2 = ax1.legend([v1,v2], ["Single prop. lost", "Detection"], loc=4, frameon=False)
# figure = plt.gcf() # get current figure
# figure.set_size_inches(16, 12)
# plt.savefig("/home/rohit/Documents/raisim_stuff/images/2_z.png", bbox_inches = 'tight',
    # pad_inches = 0, dpi=150)
# plt.tight_layout()


# plot zoomed Z
fig, ax1 = plt.subplots(1,1)
# print(pos_plot[:,2])
p1, = ax1.plot(np.arange(np.size(pos_plot[750:1400,2]), dtype=np.int32)/100 + 7.5, (pos_plot[750:1400,2]), label='actual', linewidth=3, color='#0000ff')
p2, = ax1.plot(np.arange(np.size(pos_plot[750:1400,2]), dtype=np.int32)/100 + 7.5, setpoint[750:1400,2]/2 + 5, linestyle='--', dashes=(3, 3), label='setpoint', linewidth=4, color='#008000')
v1 = ax1.axvline(x=10.0, color='black', linestyle='-.', linewidth=3, label = 'Loss of single propeller')
v2 = ax1.axvline(x=detect_ste/100, color='r', linewidth=3, label = 'Detected loss of single propeller')
# ax1.set_ylim([4.5,5.1])

# plot ang vel
fig, ax1 = plt.subplots()
# print(pos_plot[:,2])
p1, = ax1.plot(np.arange(np.size(angV_plot[:,0]), dtype=np.int32)/100 , (angV_plot[:,0]), color = '#0000ff', linewidth=3)
p2, = ax1.plot(np.arange(np.size(angV_plot[:,1]), dtype=np.int32)/100 , (angV_plot[:,1]), linestyle='--', dashes=(1,1), color = '#008000', linewidth=3)
p3, = ax1.plot(np.arange(np.size(angV_plot[:,2]), dtype=np.int32)/100 , (angV_plot[:,2]), linestyle='-.', color = '#FF00FF', linewidth=3)
v1 = ax1.axvline(x=10.0, color='black', linestyle='-.', linewidth=3, label = 'Loss of single propeller')
v2 = ax1.axvline(x=detect_ste/100, color='r', linewidth=3, label = 'Detected loss of single propeller')
ax1.set_ylabel("Angular Vel [rad/s]")
ax1.set_xlabel("Timestep [sec]")
# ax1.set_ylim([-0.5,0.5]) 
l1 = ax1.legend([p1,p2,p3], [r'$\omega_{x}$',r'$\omega_{y}$',r'$\omega_{z}$'], frameon=False, loc=2)
# ax1.add_artist(l1)
# l2 = ax1.legend([v1,v2], ["Single prop. lost", "Detection"], loc=4, frameon=False)
# figure = plt.gcf() # get current figure
# figure.set_size_inches(16, 12)
# plt.savefig("/home/rohit/Documents/raisim_stuff/images/2_pqr.png", bbox_inches = 'tight',
#     pad_inches = 0, dpi=150)
# plt.tight_layout()

if(args.plot == 1):
    plt.show()