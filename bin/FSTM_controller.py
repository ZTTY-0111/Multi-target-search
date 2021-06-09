#!/usr/bin/env python 
import os,sys
import time
import numpy as np
import torch
import random
import torch.optim
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def model(set_time, w):
    #landmark_pos = []
    #for i in range(0,16):
        #position = np.random.uniform(-2, 2, 2)
        #landmark_pos.append(position) 
    #landmark_pos = [[-1.5,0.5],[0,1],[1.5,0.9],[-0.8,-0.9],[1.3,0.1],[-0.5,1.4],[0.2,-1],[0.6,-1.5],[1.11,-0.21],[-0.3,-1.3],[0.1,0.5],[1.1,-0.7],[-1.1,0.11],[-0.4,-1.21],[-1.3,-0.2],[0.5,-0.4]]
    landmark_pos = [[-2,0.5],[3,2.5],[1.5,2],[-0.8,-0.9],[2.5,1.8],[-2.2,1.4],[2.6,-2.4],[-1.7,2.3],[1.11,-0.21],[-2,-1.6],[2.3,1.5],[1.1,-0.7],[-1.1,0.11],[-0.4,-1.21],[-1.3,-0.2],[2.4,-0.4]]
    runtime = 0
    banch_time = [0,0,0,0,0,0,0,0,0,0]
    banch = 0
    long_time = 2000
    while banch < 10:
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
        args = parser.parse_args()
     
        # load scenario from script
        scenario = scenarios.load(args.scenario).Scenario()
        # create world
        world = scenario.make_world_PFSMS(landmark_pos,set_time, w)
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
        # create interactive policies for each agent
        #policies = [InteractivePolicy(env,i) for i in range(env.n)]
        #execution loop
        obs_n = env.reset(landmark_pos)
        while True:
            runtime += 1
            end = True
            # query for action from each agent's policy
            act_n = []
            act_m = []
            for i, agent in enumerate(env.world.agents):
                act_n.append(env.imp_action(agent, obs_n[i],env._get_reward(agent)))
                agent.history_state[agent.state_place] = [agent.state.p_pos, env._get_reward(agent)]
                agent.state_place = (agent.state_place + 1)%10
            # step environment
            obs_n, done_n, _ = env.step(act_n)
            #for i, landmark in enumerate(env.world.landmarks):
                #act_m.append(env.action_landmark(landmark, obs_m[i],env._get_reward(landmark)))
            #obs_m, done_m, _ = env.step(act_m)
            # render all agent views
            env.render()
            # display rewards
            for l in env.world.landmarks:
                if l.handle > 0 and l.is_obstacle == False:
                    end = False
                    break
            if runtime == long_time:
                print("failed")
                break
            if end:
                print("all target has been handled!!!!")
                print(runtime)
                break
            #for agent in env.world.agents:
                #print(agent.name + " distance: %0.3f" % env._get_reward(agent))
        if banch < 10:
            banch_time[banch] = runtime
        if banch == 9:
            Sum = 0
            for x in banch_time:
                Sum = Sum + x
            Sum = Sum/10.0
            print("average value:")
            print(Sum)   
        banch += 1
        runtime = 0
    return Sum
'''
param_grid = {
    'set_time': list(range(25, 70)),
    'w': list(np.logspace(np.log10(0.6), np.log10(0.9), base = 10, num = 500 )),
}
MAX_EVALS = 100
best_score = 10000
best_hyperparams = {}

if __name__ == '__main__':
    for i in range(MAX_EVALS):
        print("This is the" , i)
        random.seed(i)  # 设置随机种子，每次搜索设置不同的种子，若种子固定，那每次选取的超参都是一样的
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        set_time = hyperparameters['set_time']
        w = hyperparameters['w']

        Sum = 0
        for i in range(1,4):
            score = model(set_time, w)
            Sum = Sum + score 
        score = Sum/3.0    

        if score < best_score:
            best_hyperparams = hyperparameters
            best_score = score
        print("now is ",best_score)
        print("nember is",best_hyperparams)

    print(best_score)
    print(best_hyperparams)'''
if __name__ == '__main__':
        set_time = 90
        w = random.uniform(0.6,0.9)
        score = model(set_time, w)
        print(score)