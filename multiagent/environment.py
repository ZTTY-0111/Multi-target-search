import gym
import random
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.landmarks = self.world.landmarks
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        u = np.zeros(5)
        # set action for each landmark
        for i, landmark in enumerate(self.landmarks):
            if landmark.is_obstacle:
                break
            u[1] = 0#random.uniform(0, 0.1)
            u[2] = 0#random.uniform(0, 0.1)
            u[3] = 0#random.uniform(0, 0.1)
            u[4] = 0#random.uniform(0, 0.1)
            action_m = np.concatenate([u, np.zeros(self.world.dim_c)])
            #self._set_action_landmark(action_m, landmark)
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))
        # advance world state

        return obs_n, done_n, info_n

    def _set_action_landmark(self, action, landmark, action_space, time=None):
        landmark.action.u = np.zeros(self.world.dim_p)
        landmark.action.c = np.zeros(self.world.dim_c)
        


        if landmark.movable:
            # physical action
            if self.discrete_action_input:
                landmark.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: landmark.action.u[0] = -1.0
                if action[0] == 2: landmark.action.u[0] = +1.0
                if action[0] == 3: landmark.action.u[1] = -1.0
                if action[0] == 4: landmark.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    landmark.action.u[0] += action[0][1] - action[0][2]
                    landmark.action.u[1] += action[0][3] - action[0][4]
                else:
                    landmark.action.u = action[0]
            sensitivity = 5.0
            if landmark.accel is not None:
                sensitivity = landmark.accel
            landmark.action.u *= sensitivity
            action = action[1:]
    def reset(self, landmark_pos):
        # reset world
        self.reset_callback(self.world, landmark_pos)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n   
   
    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            #print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx
#——————————————————————————————————————A Probabilistic Finite State Machine based Strategy ————————————————————————————————————————#
    #check the number of neighbor and whether it find a tagret or not
    def find_neighbor(self, agent):
        num = 0
        agent1 = None
        agent2 = None
        target_pos = None
        neighbor_state = [0,0,0,0]
        state_quadrant = 0
        temp_num = 100
        for a in self.world.agents:
            if a.alive :
                dists = np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
                if dists < 1 :
                    if a.near_false_target != -1:
                        not_record = True
                        for place in agent.false_target_state:
                            if place == -1:
                                break
                            if a.near_false_target == place:
                                not_record = False
                                break
                        if not_record:
                            agent.false_target_state[agent.false_target_place] = a.near_false_target
                            agent.false_target_place = (agent.false_target_place + 1)%5
                    x1 = a.state.p_pos[0] - agent.state.p_pos[0]
                    y1 = a.state.p_pos[1] - agent.state.p_pos[1]
                    if x1 > 0 and y1 > 0:
                        neighbor_state[0] += 1
                    elif x1 < 0 and y1 > 0:
                        neighbor_state[1] += 1
                    elif x1 < 0 and y1 < 0:
                        neighbor_state[2] += 1
                    elif x1 > 0 and y1 < 0:
                        neighbor_state[3] += 1
                    #record the best and worst agent
                    if agent1 == None and agent2 == None:
                        agent1 = a
                        agent2 = a
                    else:
                        if self._get_reward(agent1) > self._get_reward(a):
                            agent1 = a
                        elif self._get_reward(agent2) < self._get_reward(a):
                            agent2 = a  
                    num += 1
                    if a.find_target : 
                        target_pos = a.find_target_pos
        #consider that the distance less than 0.3 mains get the target
        for i in range(4):
            if neighbor_state[i] < temp_num:
                temp_num = neighbor_state[i]
                state_quadrant = i
        return target_pos, num, agent1, agent2, state_quadrant

    #Triangle Gradient Estimation
    def TriangleGradientEstimation(self, num, P1, P2, P3, f1, f2, f3, last_v):
        u = np.zeros(5) 
        a0 = 0
        b0 = 0
        if f1 > f2:
            temp_P = P1
            P1 = P2
            P2 = temp_P
            temp_f = f1
            f1 = f2
            f2 = temp_f
        if f1 > f3:
            temp_P = P1
            P1 = P3
            P3 = temp_P
            temp_f = f1
            f1 = f3
            f3 = temp_f
        if f2 > f3:
            temp_P = P2
            P2 = P3
            P3 = temp_P
            temp_f = f2
            f2 = f3
            f3 = temp_f
        if f1 == f3 and f1 == f2:
            v = last_v
            xT = v[1] - v[2]
            yT = v[3] - v[4]
            if xT > 0:
                a0 = 1
                u[1] = xT
            else:
                a0 = 2
                u[2] = -xT
            if yT > 0:
                b0 = 3
                u[3] = yT
            else:
                b0 = 4
                u[4] = -yT
        elif f1 == f2 and f1 != f3:
            x = (P1[0] + P2[0])/2.0 - P3[0]
            if x > 0:
                u[1] = x
                a0 = 1
            else:
                u[2] = -x
                a0 = 2
            y = (P1[1] + P2[1])/2.0 - P3[1]
            if y > 0:
                u[3] = y
                b0 = 3
            else:
                u[4] = -y
                b0 = 4
        elif f2 == f3:
            x = P1[0] - (P2[0] + P3[0])/2.0
            if x > 0:
                u[1] = x
                a0 = 1
            else:
                u[2] = -x
                a0 = 2
            y = P1[1] - (P2[1] + P3[1])/2.0
            if y > 0:
                u[3] = y
                b0 = 3
            else:
                u[4] = -y
                b0 = 4
        else:
            PB = [0,0]
            PB[0] = P1[0] + (P3[0] - P1[0]) * (f2 - f1)/(f3 - f1)
            PB[1] = P1[1] + (P3[1] - P1[1]) * (f2 - f1)/(f3 - f1)
            PB[0] = PB[0] - P2[0]
            PB[1] = PB[1] - P2[1]
            x = PB[1]
            y = -PB[0]
            if x * (P1[0] - P3[0]) + y * (P1[1] - P3[1]) < 0:
                x = -x
                y = -y
            if x > 0:
                u[1] = x
                a0 = 1
            else:
                u[2] = -x
                a0 = 2
            #y = P1[1] - (P2[1] + P3[1])/2
            if y > 0:
                u[3] = y
                b0 = 3
            else:
                u[4] = -y
                b0 = 4
        #control the speed  
        Theoretical_v = np.sqrt(u[a0] * u[a0] + u[b0] * u[b0])
        if Theoretical_v > 0.1:
            u[a0] = u[a0] * 0.1 / Theoretical_v
            u[b0] = u[b0] * 0.1 / Theoretical_v

        if num > 1:
            x_random = random.uniform(-0.1,0.1)
            y_random = np.sqrt(0.01 - x_random * x_random) * random.choice([1,-1])
            u[a0] = 0.9 * u[a0] + 0.1 * x_random
            u[b0] = 0.9 * u[b0] + 0.1 * y_random
        else: 
            w = 0.55
            u[1] = w * last_v[1] + (1 - w) * u[1]
            u[2] = w * last_v[2] + (1 - w) * u[2]
            u[3] = w * last_v[3] + (1 - w) * u[3]
            u[4] = w * last_v[4] + (1 - w) * u[4]
        return u
            

    # do the triangle search
    def triangle_search(self, num, agent, agent1, agent2, reward):
        P1 = agent.state.p_pos
        f1 = reward
        if num > 1:
            P2 = agent1.state.p_pos
            f2 = self._get_reward(agent1)
            P3 = agent2.state.p_pos
            f3 = self._get_reward(agent2)
        elif num == 1:
            P2 = agent1.state.p_pos
            f2 = self._get_reward(agent1)
            f3 = 1000
            for history in agent.history_state:
                if history == 0:
                   break
                if history[1] < f3:
                   f3 = history[1]
                   P3 = history[0]
        elif num == 0:
            f2 = 1000
            f3 = 0
            for history in agent.history_state:
                if history == None:
                   break
                if history[1] < f2:
                   f2 = history[1]
                   P2 = history[0]
                if history[1] > f3:
                   f3 = history[1]
                   P3 = history[0]

        v = self.TriangleGradientEstimation(num, P1, P2, P3, f1, f2, f3, agent.last_velocity)
        #if num > 1:
            #x_random = random.uniform(-0.1,0.1)
            #y_random = [np.sqrt(0.01 - x_random * x_random)] * random.choice([1,-1])
            #v[x] = 0.9*float(v[x]) + 0.1 * x_random
            #v[y] = 0.9*float(v[y]) + 0.1 * y_random
        #else: 
            #last_v = agent.last_velocity
            #w = random.uniform(0,1)
            #v[x] = w * float(last_v[x]) + (1 - w) * float(v[x])
            #v[y] = w * float(last_v[y]) + (1 - w) * float(v[y])
        return v
                   
    # # control the agent's action
    # def action(self, agent, obs, reward):
    #     u = np.zeros(5) 
    #     #what can I do after handle agent.keep_state = 0
    #     #if it's not alive then it can't move any more
    #     if agent.alive == True:
    #         #<1>find target
    #         if reward <= 5:
    #             agent.find_target_landmark.handle -= 1
    #             agent.keep_state = 0
    #             agent.keep_state_time = 0
    #             u[0] += 1.0
    #             agent.last_velocity = u
    #             return np.concatenate([u, np.zeros(self.world.dim_c)])
    #         #check the neighbor
    #         target_pos, num, agent1, agent2, quadrant = self.find_neighbor(agent)
    #         #improve the initial state
    #         if agent.initial_time:
    #             agent.initial_time = False
    #             if quadrant == 0:
    #                 u[1] = random.uniform(0, 0.1)
    #                 u[3] = np.sqrt(0.01 - u[1] * u[1])
    #             elif quadrant == 1:
    #                 u[2] = random.uniform(0, 0.1)
    #                 u[3] = np.sqrt(0.01 - u[2] * u[2])
    #             elif quadrant == 2:
    #                 u[2] = random.uniform(0, 0.1)
    #                 u[4] = np.sqrt(0.01 - u[2] * u[2])
    #             elif quadrant == 3:
    #                 u[1] = random.uniform(0, 0.1)
    #                 u[4] = np.sqrt(0.01 - u[1] * u[1])
    #             agent.last_velocity = u
    #             return np.concatenate([u, np.zeros(self.world.dim_c)])
    #         #<2>neighbor find the target
    #         if target_pos != None:
    #             if agent.state.p_pos[0] < target_pos[0]:
    #                 u[1] = target_pos[0] - agent.state.p_pos[0]
    #                 x_2 = 1 
    #             else:
    #                 u[2] = agent.state.p_pos[0] - target_pos[0]
    #                 x_2 = 2
    #             if agent.state.p_pos[1] < target_pos[1]:
    #                 u[3] = target_pos[1] - agent.state.p_pos[1]
    #                 y_2 = 3
    #             else:
    #                 u[4] = agent.state.p_pos[1] - target_pos[1]
    #                 y_2 = 4           
    #             #control the speed  
    #             Theoretical_v = np.sqrt(u[x_2] * u[x_2] + u[y_2] * u[y_2])
    #             if Theoretical_v > 0.1:
    #                 u[x_2] = u[x_2] * 0.1 / Theoretical_v
    #                 u[y_2] = u[y_2] * 0.1 / Theoretical_v
    #             agent.last_velocity = u
    #             return np.concatenate([u, np.zeros(self.world.dim_c)])
    #         # whether R1 < Ph or not
    #         R1 = random.uniform(0,1)
    #         if R1 < agent.Ph:
    #             agent.Ph = agent.Ph * 0.9997
    #             #diffusion
    #             if agent.keep_state == 2:
    #                 agent.keep_state_time += 1
    #                 u = agent.last_velocity
    #                 return np.concatenate([u, np.zeros(self.world.dim_c)])
    #             #search
    #             else: 
    #                 u = self.triangle_search(num, agent, agent1, agent2, reward)
    #                 agent.keep_state_time += 1
    #                 agent.last_velocity = u
    #                 return np.concatenate([u, np.zeros(self.world.dim_c)])
    #         #<4> wherther R1<Pd or not
    #         R2 = random.uniform(0,1)
    #         #calculate Pd
    #         if agent.keep_state_time < 2.3:
    #             Pd = 0
    #         else:
    #             Pd = 1 - 2.3/agent.keep_state_time
    #         if R2 < Pd:
    #             agent.Ph = 0.9997
    #             agent.keep_state = 2
    #             agent.keep_state_time = 0
    #             #choose a direction with less neighbor
    #             if quadrant == 0:
    #                 u[1] = random.uniform(0, 0.1)
    #                 u[3] = np.sqrt(0.01 - u[1] * u[1])
    #             elif quadrant == 1:
    #                 u[2] = random.uniform(0, 0.1)
    #                 u[3] = np.sqrt(0.01 - u[2] * u[2])
    #             elif quadrant == 2:
    #                 u[2] = random.uniform(0, 0.1)
    #                 u[4] = np.sqrt(0.01 - u[2] * u[2])
    #             elif quadrant == 3:
    #                 u[1] = random.uniform(0, 0.1)
    #                 u[4] = np.sqrt(0.01 - u[1] * u[1])
    #             agent.last_velocity = u
    #             return np.concatenate([u, np.zeros(self.world.dim_c)])
    #         else:
    #             agent.Ph = 0.9997
    #             agent.keep_state = 1
    #             agent.keep_state_time = 0
    #             u = self.triangle_search(num, agent, agent1, agent2, reward)
    #             agent.last_velocity = u
    #             return np.concatenate([u, np.zeros(self.world.dim_c)])
    #         agent.last_velocity = u
    #         return np.concatenate([u, np.zeros(self.world.dim_c)])

    # control the agent's action
    def control_speed(self, x, y):
        U = np.zeros(5)
        if x > 0:
            a = 1
            U[1] = x
        else:
            a = 2
            U[2] = -x
        if y > 0:
            b = 3
            U[3] = y
        else:
            b = 4
            U[4] = -y
        return U, a, b

    def imp_action(self, agent, obs, reward):
        u = np.zeros(5) 
        #what can I do after handle agent.keep_state = 0
        #if it's not alive then it can't move any more
        if agent.alive == True:
            #<1>find target
            if reward <= 3:
                if agent.find_target_landmark.is_obstacle:
                    #check if the target is false target
                    not_record = True
                    for place in agent.false_target_state:
                        if place == -1:
                            break
                        if agent.find_target_landmark.id == place:
                            not_record = False
                            break
                    if not_record:
                        agent.near_false_target = agent.find_target_landmark.id
                        agent.false_target_state[agent.false_target_place] = agent.find_target_landmark.id
                        agent.false_target_place = (agent.false_target_place + 1)%5
                    u[0] += 1.0
                    agent.last_velocity = u
                    return np.concatenate([u, np.zeros(self.world.dim_c)])
                agent.meet_target_time = agent.set_time
                agent.return_time = 0
                agent.meet_target_pos = agent.find_target_landmark.state.p_pos
                agent.find_target_landmark.handle -= 1
                agent.keep_state = 0
                agent.keep_state_time = 0
                u[0] += 1.0
                agent.last_velocity = u
                return np.concatenate([u, np.zeros(self.world.dim_c)])
            #check the neighbor
            target_pos, num, agent1, agent2, quadrant = self.find_neighbor(agent)
            #improve the initial state
            '''
            if agent.initial_time:
                agent.meet_target_pos = agent.state.p_pos
                agent.initial_time = False
                if quadrant == 0:
                    u[1] = random.uniform(0, 0.1)
                    u[3] = np.sqrt(0.01 - u[1] * u[1])
                elif quadrant == 1:
                    u[2] = random.uniform(0, 0.1)
                    u[3] = np.sqrt(0.01 - u[2] * u[2])
                elif quadrant == 2:
                    u[2] = random.uniform(0, 0.1)
                    u[4] = np.sqrt(0.01 - u[2] * u[2])
                elif quadrant == 3:
                    u[1] = random.uniform(0, 0.1)
                    u[4] = np.sqrt(0.01 - u[1] * u[1])
                agent.last_velocity = u
                return np.concatenate([u, np.zeros(self.world.dim_c)])'''
            #<2>neighbor find the target
            if target_pos != None:
                agent.meet_target_time = agent.set_time
                agent.return_time = 0
                if agent.state.p_pos[0] < target_pos[0]:
                    u[1] = target_pos[0] - agent.state.p_pos[0]
                    x_2 = 1 
                else:
                    u[2] = agent.state.p_pos[0] - target_pos[0]
                    x_2 = 2
                if agent.state.p_pos[1] < target_pos[1]:
                    u[3] = target_pos[1] - agent.state.p_pos[1]
                    y_2 = 3
                else:
                    u[4] = agent.state.p_pos[1] - target_pos[1]
                    y_2 = 4           
                #control the speed  
                Theoretical_v = np.sqrt(u[x_2] * u[x_2] + u[y_2] * u[y_2])
                if Theoretical_v > 0.1:
                    u[x_2] = u[x_2] * 0.1 / Theoretical_v
                    u[y_2] = u[y_2] * 0.1 / Theoretical_v
                agent.last_velocity = u
                return np.concatenate([u, np.zeros(self.world.dim_c)])
            agent.meet_target_time -= 1
            # whether R1 < Ph or not
            R1 = random.uniform(0,1)
            if R1 < agent.Ph:         
                agent.Ph = agent.Ph * 0.9997
                #diffusion
                if agent.keep_state == 2:
                    agent.keep_state_time += 1
                    u = agent.last_velocity
                #search
                else: 
                    u = self.triangle_search(num, agent, agent1, agent2, reward)
                    agent.keep_state_time += 1
                if agent.meet_target_time == 0:
                    #stop the return state
                    u_temp = np.zeros(5)
                    if agent.return_time == agent.set_time:
                        agent.return_time = 0
                        agent.meet_target_time = agent.set_time
                    elif reward < 10:
                        agent.return_time = 0
                        agent.meet_target_time = agent.set_time
                        agent.keep_state = 1
                        agent.keep_state_time = 0
                    else:
                        agent.return_time += 1
                        agent.keep_state = 2
                        agent.keep_state_time = 0
                        if agent.state.p_pos[0] < agent.meet_target_pos[0]:
                            u_temp[1] = agent.meet_target_pos[0] - agent.state.p_pos[0]
                            x_3 = 1 
                        else:
                            u_temp[2] = agent.state.p_pos[0] - agent.meet_target_pos[0]
                            x_3 = 2
                        if agent.state.p_pos[1] < agent.meet_target_pos[1]:
                            u_temp[3] = agent.meet_target_pos[1] - agent.state.p_pos[1]
                            y_3 = 3
                        else:
                            u_temp[4] = agent.state.p_pos[1] - agent.meet_target_pos[1]
                            y_3 = 4
                        Theoretical_v = np.sqrt(u_temp[x_3] * u_temp[x_3] + u_temp[y_3] * u_temp[y_3])
                        if Theoretical_v > 0.1:
                            u_temp[x_3] = u_temp[x_3] * 0.1 / Theoretical_v
                            u_temp[y_3] = u_temp[y_3] * 0.1 / Theoretical_v
                        wu = random.uniform(0.6,0.9)
                        u[1] = u_temp[1] * wu + u[1] * (1-wu)
                        u[2] = u_temp[2] * wu + u[2] * (1-wu)
                        u[3] = u_temp[3] * wu + u[3] * (1-wu)
                        u[4] = u_temp[4] * wu + u[4] * (1-wu)
                        xu = u[1] - u[2]
                        yu = u[3] - u[4]
                        u ,_ ,_ = self.control_speed(xu, yu)
                        agent.last_velocity = u
                        return np.concatenate([u, np.zeros(self.world.dim_c)])        
                agent.last_velocity = u
                return np.concatenate([u, np.zeros(self.world.dim_c)])                   
            #<4> wherther R1<Pd or not
            R2 = random.uniform(0,1)
            #calculate Pd
            if agent.keep_state_time < 2.3:
                Pd = 0
            else:
                Pd = 1 - 2.3/agent.keep_state_time
            if R2 < Pd:
                agent.Ph = 0.9997
                agent.keep_state = 2
                agent.keep_state_time = 0
                #choose a direction with less neighbor
                if quadrant == 0:
                    u[1] = random.uniform(0, 0.1)
                    u[3] = np.sqrt(0.01 - u[1] * u[1])
                elif quadrant == 1:
                    u[2] = random.uniform(0, 0.1)
                    u[3] = np.sqrt(0.01 - u[2] * u[2])
                elif quadrant == 2:
                    u[2] = random.uniform(0, 0.1)
                    u[4] = np.sqrt(0.01 - u[2] * u[2])
                elif quadrant == 3:
                    u[1] = random.uniform(0, 0.1)
                    u[4] = np.sqrt(0.01 - u[1] * u[1])
            else:
                agent.Ph = 0.9997
                agent.keep_state = 1
                agent.keep_state_time = 0
                u = self.triangle_search(num, agent, agent1, agent2, reward)
            if agent.meet_target_time == 0:
                #stop the return state
                u_temp = np.zeros(5)
                if  agent.return_time == agent.set_time:
                    agent.return_time = 0
                    agent.meet_target_time = agent.set_time
                elif reward < 10 :
                    agent.return_time = 0
                    agent.meet_target_time = agent.set_time
                    agent.keep_state = 1
                    agent.keep_state_time = 0
                else:
                    agent.return_time += 1
                    agent.keep_state = 2
                    agent.keep_state_time = 0
                    if agent.state.p_pos[0] < agent.meet_target_pos[0]:
                        u_temp[1] = agent.meet_target_pos[0] - agent.state.p_pos[0]
                        x_3 = 1 
                    else:
                        u_temp[2] = agent.state.p_pos[0] - agent.meet_target_pos[0]
                        x_3 = 2
                    if agent.state.p_pos[1] < agent.meet_target_pos[1]:
                        u_temp[3] = agent.meet_target_pos[1] - agent.state.p_pos[1]
                        y_3 = 3
                    else:
                        u_temp[4] = agent.state.p_pos[1] - agent.meet_target_pos[1]
                        y_3 = 4
                    Theoretical_v = np.sqrt(u_temp[x_3] * u_temp[x_3] + u_temp[y_3] * u_temp[y_3])
                    if Theoretical_v > 0.1:
                        u_temp[x_3] = u_temp[x_3] * 0.1 / Theoretical_v
                        u_temp[y_3] = u_temp[y_3] * 0.1 / Theoretical_v
                    wu = random.uniform(0.6,0.9)
                    u[1] = u_temp[1] * wu + u[1] * (1-wu)
                    u[2] = u_temp[2] * wu + u[2] * (1-wu)
                    u[3] = u_temp[3] * wu + u[3] * (1-wu)
                    u[4] = u_temp[4] * wu + u[4] * (1-wu)
                    xu = u[1] - u[2]
                    yu = u[3] - u[4]
                    u ,_ ,_ = self.control_speed(xu, yu)
                    agent.last_velocity = u
                    return np.concatenate([u, np.zeros(self.world.dim_c)]) 
            agent.last_velocity = u
            return np.concatenate([u, np.zeros(self.world.dim_c)])
#—————————————————————————————A Probabilistic Finite State Machine based Strategy ———————————————————————————————#

#____________________________________________________PSO_________________________________________________________#
    def get_bestReward(self, agent):
        agent.overall_best_reward = 1000
        agent.overall_best_place = None
        for nearAgent in self.agents:
            if nearAgent.group == agent.group:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - nearAgent.state.p_pos)))
                if dist <= 1:
                    if agent.overall_best_reward > self._get_reward(nearAgent):
                        agent.overall_best_reward = self._get_reward(nearAgent)
                        agent.overall_best_place = nearAgent.state.p_pos

    def PSO_action(self, agent, obs, reward):
        u = np.zeros(5) 
        #what can I do after handle agent.keep_state = 0
        #if it's not alive then it can't move any more
        if agent.alive == True:
            if reward < 3:
                if agent.find_target_landmark.is_obstacle:
                    #check if the target is false target
                    not_record = True
                    for place in agent.false_target_state:
                        if place == -1:
                            break
                        if agent.find_target_landmark.id == place:
                            not_record = False
                            break
                    if not_record:
                        agent.near_false_target = agent.find_target_landmark.id
                        agent.false_target_state[agent.false_target_place] = agent.find_target_landmark.id
                        agent.false_target_place = (agent.false_target_place + 1)%5
                    u[0] += 1.0
                    agent.last_velocity = u
                    return np.concatenate([u, np.zeros(self.world.dim_c)])
                agent.meet_target_time = agent.set_time
                agent.return_time = 0
                agent.meet_target_pos = agent.find_target_landmark.state.p_pos
                agent.find_target_landmark.handle -= 1
                u[0] += 1.0
                agent.last_velocity = u
                return np.concatenate([u, np.zeros(self.world.dim_c)])
            #check the neighbor
            target_pos, num, agent1, agent2, quadrant = self.find_neighbor(agent)
            #improve the initial state
            if agent.initial_time:
                agent.meet_target_pos = agent.state.p_pos
                agent.initial_time = False
                agent.group = quadrant
                if quadrant == 0:
                    u[1] = random.uniform(0, 0.1)
                    u[3] = np.sqrt(0.01 - u[1] * u[1])
                elif quadrant == 1:
                    u[2] = random.uniform(0, 0.1)
                    u[3] = np.sqrt(0.01 - u[2] * u[2])
                elif quadrant == 2:
                    u[2] = random.uniform(0, 0.1)
                    u[4] = np.sqrt(0.01 - u[2] * u[2])
                elif quadrant == 3:
                    u[1] = random.uniform(0, 0.1)
                    u[4] = np.sqrt(0.01 - u[1] * u[1])
                agent.last_velocity = u
                return np.concatenate([u, np.zeros(self.world.dim_c)])
            w = 0.6
            C1 = 2
            C2 = 2
            C3 = 2
            R1 = random.uniform(0,1)
            R2 = random.uniform(0,1)
            R3 = random.uniform(0,1)
            last_u = agent.last_velocity
            x1 = agent.mine_best_place[0]-agent.state.p_pos[0]
            x2 = agent.overall_best_place[0]-agent.state.p_pos[0]
            y1 = agent.mine_best_place[1]-agent.state.p_pos[1]
            y2 = agent.overall_best_place[1]-agent.state.p_pos[1]
            #control the speed
            Theoretical_v = np.sqrt(x1 * x1 + y1 * y1)
            if Theoretical_v > 0.1:
                x1 = x1 * 0.1 / Theoretical_v
                y1 = y1 * 0.1 / Theoretical_v
            Theoretical_v = np.sqrt(x2 * x2 + y2 * y2)
            if Theoretical_v > 0.1:
                x2 = x2 * 0.1 / Theoretical_v
                y2 = y2 * 0.1 / Theoretical_v
            x0 = C1*R1*(x1) + C2*R2*(x2)
            y0 = C1*R1*(y1) + C2*R2*(y2)
            if x0 > 0:
                u[1] = x0
            else:
                u[2] = -x0
            if y0 > 0:
                u[3] = y0
            else:
                u[4] = -y0
            vT = np.zeros(5) 
            if quadrant == 0:
                vT[1] = random.uniform(0, 0.1)
                vT[3] = np.sqrt(0.01 - vT[1] * vT[1])
            elif quadrant == 1:
                vT[2] = random.uniform(0, 0.1)
                vT[3] = np.sqrt(0.01 - vT[2] * vT[2])
            elif quadrant == 2:
                vT[2] = random.uniform(0, 0.1)
                vT[4] = np.sqrt(0.01 - vT[2] * vT[2])
            elif quadrant == 3:
                vT[1] = random.uniform(0, 0.1)
                vT[4] = np.sqrt(0.01 - vT[1] * vT[1])
            u[1] = w*last_u[1] + u[1] + C3*R3*vT[1]
            u[2] = w*last_u[2] + u[2] + C3*R3*vT[2]
            u[3] = w*last_u[3] + u[3] + C3*R3*vT[3]
            u[4] = w*last_u[4] + u[4] + C3*R3*vT[4]
            x_random = random.uniform(-0.1,0.1)
            y_random = np.sqrt(0.01 - x_random * x_random) * random.choice([1,-1])
            xT = u[1] - u[2] + x_random
            yT = u[3] - u[4] + y_random
            v = np.zeros(5) 
            if xT > 0:
                a1 = 1
                v[1] = xT
            else:
                a1 = 2
                v[2] = -xT
            if yT > 0:
                b1 = 3
                v[3] = yT
            else:
                b1 = 4
                v[4] = -yT
            Theoretical_v = np.sqrt(v[a1] * v[a1] + v[b1] * v[b1])
            if Theoretical_v > 0.1:
                v[a1] = v[a1] * 0.1 / Theoretical_v
                v[b1] = v[b1] * 0.1 / Theoretical_v

            agent.last_velocity = v
            return np.concatenate([v, np.zeros(self.world.dim_c)])

#________________________________________________PSO_____________________________________________#

#________________________________________________GES______________________________________________#

    def Team_members(self, agent):
        best_agent = agent
        second_agent = agent
        Sum = [agent.state.p_pos[0], agent.state.p_pos[1]]
        num = 1

        for a in self.world.agents:
            if a.alive :
                dists = np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
                if dists < 1 :
                    if a.near_false_target != -1:
                        not_record = True
                        for place in agent.false_target_state:
                            if place == -1:
                                break
                            if a.near_false_target == place:
                                not_record = False
                                break
                        if not_record:
                            agent.false_target_state[agent.false_target_place] = a.near_false_target
                            agent.false_target_place = (agent.false_target_place + 1)%5
                    x = a.state.p_pos[0]
                    y = a.state.p_pos[1]
                    Sum[0] = Sum[0] + x
                    Sum[1] = Sum[1] + y
                    num = num + 1
                    if self._get_reward(best_agent) > self._get_reward(a):
                        best_agent = a
                    elif self._get_reward(second_agent) > self._get_reward(a):
                        second_agent = a
        Sum[0] = Sum[0]/num
        Sum[1] = Sum[1]/num
        num = num - 1
        return Sum, best_agent ,second_agent, num

    def GES_action(self, agent, obs, reward):
        u = np.zeros(5) 
        #what can I do after handle agent.keep_state = 0
        #if it's not alive then it can't move any more
        if agent.alive == True:
            #find the target
            if reward < 3:
                if agent.find_target_landmark.is_obstacle:
                    #check if the target is false target
                    not_record = True
                    for place in agent.false_target_state:
                        if place == -1:
                            break
                        if agent.find_target_landmark.id == place:
                            not_record = False
                            break
                    if not_record:
                        agent.near_false_target = agent.find_target_landmark.id
                        agent.false_target_state[agent.false_target_place] = agent.find_target_landmark.id
                        agent.false_target_place = (agent.false_target_place + 1)%5
                    agent.last_velocity = u
                    return np.concatenate([u, np.zeros(self.world.dim_c)])
                agent.find_target_landmark.handle -= 1
                agent.last_velocity = u
                return np.concatenate([u, np.zeros(self.world.dim_c)])

            #improve the initial state
            if agent.initial_time:
                target_pos, num, agent1, agent2, quadrant = self.find_neighbor(agent)
                agent.initial_time = False
                if quadrant == 0:
                    u[1] = random.uniform(0, 0.1)
                    u[3] = np.sqrt(0.01 - u[1] * u[1])
                elif quadrant == 1:
                    u[2] = random.uniform(0, 0.1)
                    u[3] = np.sqrt(0.01 - u[2] * u[2])
                elif quadrant == 2:
                    u[2] = random.uniform(0, 0.1)
                    u[4] = np.sqrt(0.01 - u[2] * u[2])
                elif quadrant == 3:
                    u[1] = random.uniform(0, 0.1)
                    u[4] = np.sqrt(0.01 - u[1] * u[1])
                agent.last_velocity = u
                return np.concatenate([u, np.zeros(self.world.dim_c)])
            #check the team members
            center_position, best_agent, second_agent, members_num = self.Team_members(agent)

            G = np.zeros(5)
            BetaG = 5 
            Beta = 0.5
            Rs = random.choice([1, 1 - Beta, 1 + Beta])
            if members_num < BetaG:
                Xg = (best_agent.state.p_pos[0] - center_position[0]) * Rs
                Yg = (best_agent.state.p_pos[1] - center_position[1]) * Rs
                G, ag, bg = self.control_speed(Xg, Yg)
            else:
                if agent.id == best_agent.id:
                    if agent.state.p_pos[0] < second_agent.state.p_pos[0]:
                        G[2] = second_agent.state.p_pos[0] - agent.state.p_pos[0]
                        ag = 2 
                    else:
                        G[1] = agent.state.p_pos[0] - second_agent.state.p_pos[0]
                        ag = 1
                    if agent.state.p_pos[1] < second_agent.state.p_pos[1]:
                        G[4] = second_agent.state.p_pos[1] - agent.state.p_pos[1]
                        bg = 4
                    else:
                        G[3] = agent.state.p_pos[1] - second_agent.state.p_pos[1]
                        bg = 3

                elif agent.id == second_agent.id:
                    if agent.state.p_pos[0] < best_agent.state.p_pos[0]:
                        G[2] = best_agent.state.p_pos[0] - agent.state.p_pos[0]
                        ag = 2 
                    else:
                        G[1] = agent.state.p_pos[0] - best_agent.state.p_pos[0]
                        ag = 1
                    if agent.state.p_pos[1] < best_agent.state.p_pos[1]:
                        G[4] = best_agent.state.p_pos[1] - agent.state.p_pos[1]
                        bg = 4
                    else:
                        G[3] = agent.state.p_pos[1] - best_agent.state.p_pos[1]
                        bg = 3

                else:
                    BetaP = 1
                    W1 = (self._get_reward(second_agent) - self._get_reward(best_agent))/10.0 + BetaP
                    W2 = BetaP
                    P1 = W1/(W1 + W2)
                    P2 = W2/(W1 + W2)

                    x = random.uniform(0, 1)
                    cumulative_probability = 0.0
                    for item, item_probability in zip([1,2], [P1, P2]):
                        cumulative_probability += item_probability
                        if x < cumulative_probability:
                            break

                    if item == 1:
                        if best_agent.state.p_pos[0] < second_agent.state.p_pos[0]:
                            G[2] = second_agent.state.p_pos[0] - best_agent.state.p_pos[0]
                            ag = 2 
                        else:
                            G[1] = best_agent.state.p_pos[0] - second_agent.state.p_pos[0]
                            ag = 1
                        if best_agent.state.p_pos[1] < second_agent.state.p_pos[1]:
                            G[4] = second_agent.state.p_pos[1] - best_agent.state.p_pos[1]
                            bg = 4
                        else:
                            G[3] = best_agent.state.p_pos[1] - second_agent.state.p_pos[1]
                            bg = 3
                        Xg = (best_agent.state.p_pos[0] - agent.state.p_pos[0]) * Rs + G[1] - G[2]
                        Yg = (best_agent.state.p_pos[1] - agent.state.p_pos[1]) * Rs + G[3] - G[4]
                        G, ag, bg = self.control_speed(Xg, Yg) 
                    else:
                        if best_agent.state.p_pos[0] < second_agent.state.p_pos[0]:
                            G[1] = second_agent.state.p_pos[0] - best_agent.state.p_pos[0]
                            ag = 1 
                        else:
                            G[2] = best_agent.state.p_pos[0] - second_agent.state.p_pos[0]
                            ag = 2
                        if best_agent.state.p_pos[1] < second_agent.state.p_pos[1]:
                            G[3] = second_agent.state.p_pos[1] - best_agent.state.p_pos[1]
                            bg = 3
                        else:
                            G[4] = best_agent.state.p_pos[1] - second_agent.state.p_pos[1]
                            bg = 4
                        Xg = (second_agent.state.p_pos[0] - agent.state.p_pos[0]) * Rs + G[1] - G[2]
                        Yg = (second_agent.state.p_pos[1] - agent.state.p_pos[1]) * Rs + G[3] - G[4]
                        G, ag, bg = self.control_speed(Xg, Yg) 


            best_history_state = agent.state.p_pos
            best_reward = self._get_reward(agent)
            not_zero = False
            for history in agent.history_state:
                if history == None:
                   break
                if history[1] < best_reward:
                    best_reward = history[1]
                    best_history_state = history[0]
                    not_zero = True
                elif history[1] == best_reward:
                    dist1 = np.sqrt(np.sum(np.square(history[0] - agent.state.p_pos)))
                    dist2 = np.sqrt(np.sum(np.square(best_history_state - agent.state.p_pos)))
                    if dist1 > dist2:
                        best_reward = history[1]
                        best_history_state = history[0]
                        not_zero = True
            H = np.zeros(5)
            if not_zero:
                r = random.uniform(0.4,0.8)
                Xh = (best_agent.state.p_pos[0] - best_history_state[0]) * r
                Yh = (best_agent.state.p_pos[1] - best_history_state[1]) * r
                H, ah, bh = self.control_speed(Xh, Yh)

            u = np.zeros(5)
            #the final speed
            abs_G = np.sqrt(G[ag] * G[ag] + G[bg] * G[bg])
            if not_zero:
                abs_H = np.sqrt(H[ah] * H[ah] + H[bh] * H[bh])
            else:
                abs_H = 0
            if abs_G > 0:
                u[1] = G[1] + H[1]
                u[2] = G[2] + H[2]
                u[3] = G[3] + H[3]
                u[4] = G[4] + H[4]
                Xu = u[1] - u[2]
                Yu = u[3] - u[4]
                u = np.zeros(5)
                u, au, bu = self.control_speed(Xu, Yu)
                Theoretical_v = np.sqrt(u[au] * u[au] + u[bu] * u[bu])
                if Theoretical_v > 0.1:
                    u[au] = u[au] * 0.1 / Theoretical_v
                    u[bu] = u[bu] * 0.1 / Theoretical_v
            elif abs_G == 0 and abs_H > 0:
                random_x = random.uniform(-0.01,0.01)
                random_y = np.sqrt(0.0001 - x_random * x_random) * random.choice([1,-1])
                u[ah] = H[ah] + random_x
                u[bh] = H[bh] + random_y
                Theoretical_v = np.sqrt(u[ah] * u[ah] + u[bh] * u[bh])
                if Theoretical_v > 0.1:
                    u[ah] = u[ah] * 0.1 / Theoretical_v
                    u[bh] = u[bh] * 0.1 / Theoretical_v
            else:
                u = agent.last_velocity
            agent.last_velocity = u
            return np.concatenate([u, np.zeros(self.world.dim_c)])
    def test_action(self, agent, obs, reward):
        u = np.zeros(5)
        u[0] += 1
        return np.concatenate([u, np.zeros(self.world.dim_c)])