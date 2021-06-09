import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world_PFSMS(self, landmark_pos, set_time, w):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 50
        num_landmarks = 10
        num_obstacles =6
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.03
            agent.alive = True
            agent.set_time = set_time
            agent.w = w
            agent.meet_target_time = agent.set_time
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks + num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            if(i < num_landmarks):
                landmark.name = 'landmark %d' % i
                landmark.id = i
                landmark.collide = False
                landmark.movable = False
                landmark.handle = 100
                landmark.size = 0.05
                landmark.color = np.array([0, 0, 1])
            else:
                landmark.name = 'landmark %d' % i
                landmark.id = i
                landmark.collide = False
                landmark.movable = False
                landmark.is_obstacle = True
                landmark.handle = 100
                landmark.size = 0.05
                landmark.color = np.array([0.25, 0.25, 0.25])
        # make initial conditions
        self.reset_world(world, landmark_pos)
        return world
    def make_world(self, landmark_pos):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 50
        num_landmarks = 10
        num_obstacles =6
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.03
            agent.alive = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks + num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            if(i < num_landmarks):
                landmark.name = 'landmark %d' % i
                landmark.id = i
                landmark.collide = False
                landmark.movable = False
                landmark.handle = 100
                landmark.size = 0.05
                landmark.color = np.array([0, 0, 1])
            else:
                landmark.name = 'landmark %d' % i
                landmark.id = i
                landmark.collide = False
                landmark.movable = False
                landmark.is_obstacle = True
                landmark.handle = 100
                landmark.size = 0.05
                landmark.color = np.array([0.25, 0.25, 0.25])
        # make initial conditions
        self.reset_world(world, landmark_pos)
        return world

    def reset_world(self, world, landmark_pos):
        #landmark_pos_1 = [[-1.5,0.5],[0,1],[1.5,0.9],[-0.8,-0.9],[1.3,0.1],[-0.5,1.4],[0.2,-1],[0.6,-1.5],[1.11,-0.21],[-0.3,-1.3],[0.1,0.5],[1.1,-0.7],[-1.1,0.11],[-0.4,-1.21],[-1.3,-0.2],[0.5,-0.4]]
        #landmark_pos_3 = [[-2,0.5],[3,1],[1.5,2],[-0.8,-0.9],[2.5,1.8],[-2.2,1.4],[0.2,-2.4],[-1.7,2.3],[1.11,-0.21],[-2,-1.6],[0.1,0.5],[1.1,-0.7],[-1.1,0.11],[-0.4,-1.21],[-1.3,-0.2],[0.5,-0.4]]        
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0, 1, 0])
        # random properties for landmarks
        #for i, landmark in enumerate(world.landmarks):
            #landmark.color = np.array([0, 0, 1])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(0, 0.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            #landmark.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            landmark.state.p_pos = landmark_pos[i]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        dist = np.sqrt(np.sum(np.square(agent1.state.p_pos - agent2.state.p_pos)))
        dist_min = agent1.size + agent2.size
        print(dist, dist_min)
        #if dist < dist_min:
            #return True 
        #else: 
            #return False

    #the distance between agent to the 
    def reward(self, agent, world):
        #for a in world.agents:
            #self.is_collision(a, agent)
            #if  and a.name != agent.name:
                #a.alive = False
                #agent.alive = False
        #the range of reward 

        rew = 3
        for l in world.landmarks:
            if l.handle > 0 :
                dists = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                if rew > dists :
                    IS_obstacle = False
                    for place in agent.false_target_state:
                        if place == -1:
                            break
                        if l.id == place:
                            IS_obstacle = True
                            break
                    if IS_obstacle:
                        break
                    find_pos = l.state.p_pos
                    near_landmark = l
                    rew = dists
        #consider that the distance less than 0.3 mains get the target
        if rew <= 0.3 :
            agent.find_target = True
            agent.find_target_landmark = near_landmark
            agent.find_target_state = find_pos
        else:
            agent.find_target = False
        #out the range
        if rew > 1:
            rew = 1
        rew = rew * 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents  
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
