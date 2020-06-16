from ruamel.yaml import YAML, dump, RoundTripDumper
from raisim_gym.env.RaisimGymVecEnv import RaisimGymVecEnv as Environment
from raisim_gym.env.env.hummingbird import __HUMMINGBIRD_RESOURCE_DIRECTORY__ as __RSCDIR__
from raisim_gym.algo.ppo2 import PPO2
from raisim_gym.archi.policies import MlpPolicy, customMlpPolicy
from raisim_gym.helper.raisim_gym_helper import ConfigurationSaver, TensorboardLauncher
from _raisim_gym import RaisimGymEnv
import os
import math
import argparse
import random, math
import numpy as np
# from ann_for_prop.main_43 import main
from ann_for_prop.main_32 import main
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# K.set_session(sess)

detect_ste = -1
prop_val = 0

main = main(1e-4, train=False)

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default=os.path.abspath(__RSCDIR__ + "/default_cfg.yaml"),
                    help='configuration file')
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w4', '--weight_4prop', help='trained weight path', type=str, default='')
parser.add_argument('-w3', '--weight_3prop', help='trained weight path', type=str, default='')
parser.add_argument('-w2', '--weight_2prop', help='trained weight path', type=str, default='')
parser.add_argument('-w1', '--weight_1prop', help='trained weight path', type=str, default='')
parser.add_argument('-wann', '--weight_ann', help='trained weight path', type=str, default='')
parser.add_argument('-n', '--name', help='name path', type=str, default='')
args = parser.parse_args()
mode = args.mode
cfg_abs_path = parser.parse_args().cfg
cfg = YAML().load(open(cfg_abs_path, 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 500
env = Environment(RaisimGymEnv(__RSCDIR__, dump(cfg['environment'], Dumper=RoundTripDumper)))

weight_prop = [args.weight_4prop, args.weight_3prop, args.weight_2prop]

model_list = []
for i in range(3):
	model_list.append(PPO2.load(weight_prop[i]))


main.load_model(args.weight_ann)
obs = env.reset()
# obs = obs.reshape((obs.shape[0],1,obs.shape[1]))
print(obs.shape)
running_reward = 0.0
ep_len = 0
switch_off1 = 1000
# switch_off2 = 2500
# switch_off3 = 3750

env.start_recording_video("/home/rohit/Documents/raisim_stuff/all_model.mp4")

env_no = cfg['environment']['num_envs']

pos = np.zeros((env_no, 15,3), dtype = np.float32)
pos_plot = np.zeros((env_no, 2000,3), dtype = np.float32)
setpoint = np.zeros((env_no,2000,3), dtype = np.float32)
# setpoint[:,:,-1] = ((np.random.rand(env_no,2000))*6
for i in range(env_no):
    setpoint[i,:,-1] = (i/env_no)*2.0 - 9.0
# print(np.amax(setpoint), np.amin(setpoint)) 
# gg
angV_plot = np.zeros((env_no, 2000,3), dtype = np.float32)

final_mean = np.zeros((env_no, 1,3), dtype = np.float32)
stab_count = 0
prev = False

count = 0
prev_p = 0

obs_append = np.zeros((env_no,200,18), dtype=obs.dtype)
obs_append[:,-1,:] = obs

prop_loss_cum = np.zeros((env_no,200,1), dtype = np.float32)

# prop_loss_check = [False, False, False, False]
prop_loss_check = [False, False, False, True]

prop_loss_check = [prop_loss_check[:] for i in range(env_no)]

# def check_prop_loss(prop_loss_cum):
#     uq = np.unique(prop_loss_cum[:,50:,:]).size
#     if(uq <= 100):
#         return prop_loss_cum[:,-1,0].reshape(env_no,1,1)
#     else:
#         return np.zeros((env_no,1,1))

# faster function
def check_prop_loss(prop_loss_cum, i):
    uq = np.unique(prop_loss_cum[i,50:,:]).size
    if(uq <= 1):
        return int(prop_loss_cum[i,-1,0])
    else:
        return 0
    
# def take_action(obs, ste):
#     global detect_ste
#     action = np.zeros((env_no,4), dtype = np.float32)
#     for i in range(env_no):
#         res = [j for j, val in enumerate(prop_loss_check[i]) if val]
#         obs_append[i,:99,:] = obs_append[i,1:,:]
#         obs_append[i,-1,:] = obs[i,:]
#         if(ste>150 and len(res) == 0):
#             prop_loss = main.predict(obs_append[i,None,:,:])
#             prop_loss_cum[i,:99,:] = prop_loss_cum[i,1:,:]
#             prop_loss_cum[i,-1,:] = np.argmax(prop_loss)
#             prop_val = check_prop_loss(prop_loss_cum,i) # use model of prop loss detection

#             # if(int(prop_val[i,:,:]) == 1):
#             #     prop_loss_check[i][int(prop_val[i,:,:]-1)] = True
#             # elif(int(prop_val[i,:,:]) == 2):
#             #     prop_loss_check[i][int(prop_val[i,:,:]-1)] = True
#             # elif(int(prop_val[i,:,:]) == 3):
#             #     prop_loss_check[i][int(prop_val[i,:,:]-1)] = True
#             # elif(int(prop_val[i,:,:]) == 4):
#             #     if(detect_ste < 0):
#             #         detect_ste = ste
#             #     prop_loss_check[i][int(prop_val[i,:,:]-1)] = True
            
#             if(prop_val == 1):
#                 prop_loss_check[i][prop_val-1] = True
#             elif(prop_val == 2):
#                 prop_loss_check[i][prop_val-1] = True
#             elif(prop_val == 3):
#                 prop_loss_check[i][prop_val-1] = True
#             elif(prop_val == 4):
#                 if(detect_ste < 0):
#                     detect_ste = ste
#                 prop_loss_check[i][prop_val-1] = True

#         if(ste<switch_off1):
#             action[i,:], _ = model_list[0].predict(obs[i,:])
#         else:
#             action[i,:], _ = model_list[0].predict(obs[i,:])
#             action[i,-1] = -1.25
#             if(len(res) == 1):
#                 if(res == 0):
#                     action[i,:], _ = model_list[1].predict(obs[i,:])
#                     action[i] = np.insert(action[i], 0, -1.25, axis = 1)
#                 elif(res == 1):
#                     action[i,:], _ = model_list[1].predict(obs[i,:])
#                     action[i] = np.insert(action[i], 1, -1.25, axis = 1)
#                 elif(res == 2):
#                     action[i,:], _ = model_list[1].predict(obs[i,:])
#                     action[i] = np.insert(action[i], 2, -1.25, axis = 1)
#                 elif(res == 3):
#                     action[i,:], _ = model_list[1].predict(obs[i,:])
#                     action[i] = np.insert(action[i], 3, -1.25, axis = 1)            
#     return action

def take_action(obs, ste):
    global detect_ste
    action = np.zeros((env_no,4), dtype = np.float32) - 1.25
    for i in range(env_no):
        res = [j for j, val in enumerate(prop_loss_check[i]) if val]
        obs_append[i,:199,:] = obs_append[i,1:,:]
        obs_append[i,-1,:] = obs[i,:]
        if(ste>250 and len(res) == 1):
            prop_loss = main.predict(obs_append[i,None,:,:])
            prop_loss_cum[i,:199,:] = prop_loss_cum[i,1:,:]
            prop_loss_cum[i,-1,:] = np.argmax(prop_loss)
            prop_val = check_prop_loss(prop_loss_cum, i)
            if(prop_val == 1):
                if(detect_ste < 0):
                    detect_ste = ste
                prop_loss_check[i][2] = True
        # print(prop_loss_check)
        if(ste<switch_off1):
            action[i,:3], _ = model_list[1].predict(obs[i,:])
            # action[i] = np.insert(action[i], 3, -1.25, axis = 1)
        else:
            action[i,:3], _ = model_list[1].predict(obs[i,:])
            action[i,2] = -1.25
            # action[i] = np.insert(action[i], 3, -1.25, axis = 1)
            if(len(res) == 2):
                if(res == [2,3]):
                    action[i,:2], _ = model_list[2].predict(obs[i,:])
                    # action[i] = np.insert(action[i], 2, -1.25, axis = 1)
                    # action[i] = np.insert(action[i], 3, -1.25, axis = 1)
    # print(action)
    return action

# def take_action(obs, ste):
#     action, _ = model_list[2].predict(obs)
#     action = np.insert(action, 2, -1.25, axis = 1)
#     action = np.insert(action, 3, -1.25, axis = 1)
#     # print(action.shape)
#     return action

try:
    for ste in range(2000):
        env.wrapper.showWindow()
        if(ste%50 == 0):
            print(ste)

        action = take_action(obs, ste)
        # print(action)
        obs, reward, done, infos = env.step(action, visualize=False)
        # obs = obs.reshape((obs.shape[0],1,obs.shape[1]))
        angV_plot[:,ste,:] = obs[:,15:]
        pos_plot[:,ste,:] = obs[:,9:12] 
        pos[:,:14,:] = pos[:,1:,:]
        pos[:,-1,:] = obs[:,9:12]
        mean = np.mean(pos - setpoint[:,None,ste,:], axis=1)
        std = np.std(pos, axis=0)
        
        if(ste>100):
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
        if done.any():
            print("Episode Reward: {:.2f}".format(running_reward))
            print("Episode Length", ep_len)
            running_reward = 0.0
            ep_len = 0
        obs[:,9:12] -= setpoint[:,ste,:]
except KeyboardInterrupt:
    env.stop_recording_video()

fail_before = np.zeros((env_no,1), dtype=np.bool)
fail_after = np.zeros((env_no,1), dtype=np.bool)
count_per = 0
for i in range(env_no):
    if((pos_plot[i,:1000,-1] < -9.75).any()):
        fail_before[i,:] = True
        fail_after[i,:] = False
    elif((pos_plot[i,1000:,-1] < -9.75).any()):
        fail_after[i,:] = True
        fail_before[i,:] = False
    else:
        count_per += 1
        fail_after[i,:] = False
        fail_before[i,:] = False
# print(fail_after)
print(count_per)
print('Fail before:', np.sum(fail_before, axis=0))
print('Fail after:', np.sum(fail_after, axis=0))

fail_plot = np.zeros((10,1), dtype=np.float32)
for i in range(10):
    gap = (env_no//10)
    fail_plot[i,:] = np.sum(fail_after[i*gap:(i+1)*gap, :], axis=0)

# np.save('./pos-4-3.npy', pos_plot)
print(fail_plot)
# pos_plot = np.load('./pos-4-3.npy')
np.save('fail_plot_32_final.npy', fail_plot)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# for i in range(env_no):
#     ax.plot3D(pos_plot[i,:,0], pos_plot[i,:,1], pos_plot[i,:,2]/2.0 + 5.0);
# plt.show()
# fig, ax1 = plt.subplots()
# plt.bar(np.arange(fail_plot.shape[0]), fail_plot[:,0], 0.35)
fig, ax1 = plt.subplots()
for i in range(env_no):
    p1, = ax1.plot(np.arange(np.size(pos_plot[i,:,-1]), dtype=np.int32)/100 , (pos_plot[i,:,-1]))
plt.show()