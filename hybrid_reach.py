from rl import DDPG
import tensorflow as tf
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.backend.sim import *
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from numpy import exp
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle
from numpy import savetxt
import os

MAX_EPISODES = 1001
MAX_EP_STEPS = 300
ON_TRAIN = True#make it false if you want to test

SCENE_FILE = './scenes/scene_reinforcement_learning_env_5_RealObsDynRandom.ttt'

action_bound = [-0.02, 0.02]
POS_MIN, POS_MAX = [0.30, -0.45, 0.08], [0.75, 0.45, 0.08]   #target positions
#POS_MIN, POS_MAX = [0.40, 0.0, 0.08], [0.40, 0.0, 0.08]     #target position for scenarios
BILL_POS_MIN, BILL_POS_MAX = [0.30, -0.45, 0.20], [0.75, 0.45, 0.20]  #obstacle positions

thresh= 0.05
penalty= 4
closest = 0.15
furthest = 2.5
evalnum = 1000

dir = './models/reach/params1/'

if not os.path.exists(dir):
        os.makedirs(dir)


class ArmEnv(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = Panda()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

        self.obstacle = Shape('Cuboid')

        self.col = simGetCollisionHandle('Collision')
        self.colG = simGetCollisionHandle('CollisionG')
        self.dis0 = simGetDistanceHandle('Distance')
        self.disT = simGetDistanceHandle('Distance0')

        self.base = simGetObjectHandle('Panda')
        self.robotObjects = simGetObjectsInTree(self.base, sim_handle_all, 0)
        self.robotInitialConfig = simGetConfigurationTree(self.base)

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position

        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        b1x, b1y, b1z = self.obstacle.get_position()

        vector_target= [ax-tx, ay-ty , az-tz]

        return (self.agent.get_joint_positions() +
                self.agent.get_joint_velocities() +
                vector_target)

    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        pos_bill = list(np.random.uniform(BILL_POS_MIN, BILL_POS_MAX))

        while (((pos[0] < pos_bill[0] + closest and pos[0] > pos_bill[0] - closest) and (pos[1] < pos_bill[1] + closest and pos[1] > pos_bill[1] - closest)) or (pos[0] > pos_bill[0] + furthest or pos[0] < pos_bill[0] - furthest or pos[1] > pos_bill[1] + furthest or pos[1] < pos_bill[1] - furthest)):
           pos = list(np.random.uniform(POS_MIN, POS_MAX))

        for i in range (len(self.robotObjects)):
            simResetDynamicObject(self.robotObjects[i])
        simSetConfigurationTree(self.robotInitialConfig)

        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.obstacle.set_position(pos_bill)
        return self._get_state()

    def step(self, action):
        done = False

        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation

        dis_target = simReadDistance(self.disT)
        # Reward is negative distance to target
        reward = -dis_target

        c0 = simReadCollision(self.col)
        cG = simReadCollision(self.colG)
        if (c0):
            collision_robot = 1
        else:
            collision_robot = 0


        if (reward > -0.05):
            done=True
            reward= reward + penalty      #params42cont

        return reward, self._get_state(), done, collision_robot

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


# set env
env = ArmEnv()
s_dim = 17
a_dim = 7
a_bound = action_bound

# set RL method (continuous)
name = 'reach'

rl = DDPG(a_dim, s_dim, a_bound,ON_TRAIN, name)

def train():
    var = 2.  # control exploration
    reward_list = []
    avg_rewards = []

    col_list = np.zeros((MAX_EPISODES))
    done_list = np.zeros((MAX_EPISODES))

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        done_count= 0

        if (i%100 == 0):
            rl.save(reward_list,avg_rewards,i,dir)
            savetxt(dir + 'reward_list.csv', np.array(reward_list), delimiter=',')
            savetxt(dir + 'avg_rewards.csv', np.array(avg_rewards), delimiter=',')
            savetxt(dir + 'done_list.csv', np.array(done_list), delimiter=',')
            savetxt(dir + 'col_list.csv', np.array(col_list), delimiter=',')

        for j in range(MAX_EP_STEPS):

            a = rl.choose_action(s)
            a = np.clip(np.random.normal(a, var), *action_bound)
            r, s_, done, col = env.step(a)

            if not done:
                done_count= 0

            else:
                done_count += 1

            if done_count==30:
                done_list[i] = 1

            if col:
                col_list[i] = 1

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                rl.learn()
                var = max([var*.9999, 0.1]) #decay the action randomness

            s = s_

            if (done_count==500 or j == MAX_EP_STEPS - 1 ):
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done_count == 500 else 'done', ep_r, j))
                reward_list.append( ep_r)
                avg_rewards.append(np.mean(reward_list[-10:]))
                break

    rl.save(reward_list, avg_rewards, i ,dir)
    savetxt(dir + 'reward_list.csv', np.array(reward_list), delimiter=',')
    savetxt(dir + 'avg_rewards.csv', np.array(avg_rewards), delimiter=',')
    savetxt(dir + 'done_list.csv', np.array(done_list), delimiter=',')
    savetxt(dir + 'col_list.csv', np.array(col_list), delimiter=',')



def eval():
    reward_list = []
    x_axis = np.array(range(evalnum))
    rl.restore(dir)
    col_list = np.zeros((evalnum), dtype=bool)
    done_list = np.zeros((evalnum), dtype=bool)
    for i in range(evalnum):

        s = env.reset()
        ep_r = 0.
        done_count = 0

        for j in range(MAX_EP_STEPS):

            a = rl.choose_action(s)

            r, s_, done, col = env.step(a)

            if (col):
                col_list[i] = True

            ep_r += r
            s = s_

            if not done:
                done_count = 0

            else:
                done_count += 1

            if (done_count == 30 or j == MAX_EP_STEPS - 1):
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done_count == 30 else 'done', ep_r, j))
                reward_list.append(ep_r)
                if done_count==30:
                    done_list[i] = True
                break


    reward_list= np.array(reward_list)

    plt.plot(x_axis, reward_list, color='b')
    plt.plot(x_axis[col_list], reward_list[col_list], 'rx')
    plt.plot(x_axis[done_list], reward_list[done_list], 'go')
    plt.title('Tested on ' + str(evalnum) + ' episodes' + '\n' + 'Success rate: ' + str(
        100 * len(reward_list[done_list]) / evalnum) + '%' + 'Collusion rate: ' + str(
        100 * len(reward_list[col_list]) / evalnum) + '%')
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.savefig(dir + 'testDynamicRandom.png')


if ON_TRAIN:
    train()
else:
    eval()

print('Evaluation Done!')

env.shutdown()
