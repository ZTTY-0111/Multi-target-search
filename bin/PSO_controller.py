#!/usr/bin/env python 
import os,sys
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # create interactive policies for each agent
    #policies = [InteractivePolicy(env,i) for i in range(env.n)]
    #execution loop
    obs_n = env.reset()
    runtime = 0
    banch_time = [0,0,0,0,0,0,0,0,0,0]
    banch = 0
    while banch < 1:
        while True:
            runtime += 1
            end = True
            act_n = []
            # query for action from each agent's policy
            for i, agent in enumerate(env.world.agents):
                rewardNow = env._get_reward(agent)
                if rewardNow <= agent.mine_best_reward:
                    agent.mine_best_reward = rewardNow
                    agent.mine_best_place = agent.state.p_pos
                env.get_bestReward(agent)
                act_n.append(env.PSO_action(agent, obs_n[i], rewardNow))
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
            if end:
                print("all target has been handled!!!!")
                print(runtime)
                break
            for agent in env.world.agents:
                print(agent.name + " distance: %0.3f" % env._get_reward(agent))
        if banch < 10:
            banch_time[banch] = runtime
        if banch == 9:
            for x in banch_time:
                print(x)
        banch += 1
        runtime = 0
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
        args = parser.parse_args()
        scenario = scenarios.load(args.scenario).Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
        obs_n = env.reset()
