from rl import DDPG
import tensorflow as tf
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.backend.sim import *
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from numpy import exp
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle

MAX_EPISODES = 1001
MAX_EP_STEPS = 300
ON_TRAIN = False#make it false if you want to test

SCENE_FILE = './scenes/scene_reinforcement_learning_env_5_RealObsDynRandom.ttt'


action_bound = [-0.02, 0.02]
POS_MIN, POS_MAX = [0.30, -0.45, 0.08], [0.75, 0.45, 0.08]   #target positions
#POS_MIN, POS_MAX = [0.40, 0.0, 0.08], [0.40, 0.0, 0.08]     #target position for scenarios
BILL_POS_MIN, BILL_POS_MAX = [0.30, -0.45, 0.20], [0.75, 0.45, 0.20]  #obstacle positions

thresh= 0.25
penalty= 5
closest = 0.10
furthest = 2.5

reach_dir = './models/reach/params1/'
avoid_dir = './models/avoid/params1/'

evalnum= 25
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

        self.joint7 = simGetObjectHandle('Panda_joint7')
        self.joint6 = simGetObjectHandle('Panda_joint6')
        self.joint5 = simGetObjectHandle('Panda_joint5')
        self.joint4 = simGetObjectHandle('Panda_joint4')
        self.joint3 = simGetObjectHandle('Panda_joint3')
        self.joint2 = simGetObjectHandle('Panda_joint2')
        self.joint1 = simGetObjectHandle('Panda_joint1')

    def _get_state(self,avoid):
        # Return state containing arm joint angles/velocities & target position

        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        b1x, b1y, b1z = self.obstacle.get_position()

        vector_target= [ax-tx, ay-ty , az-tz]
        vector_obstacle = [ax - b1x, ay - b1y, az - b1z]

        if avoid:
            return (self.agent.get_joint_positions() +
                self.agent.get_joint_velocities() +
                vector_target + vector_obstacle)
        else:
            return (self.agent.get_joint_positions() +
                    self.agent.get_joint_velocities() +
                    vector_target)

    def reset(self,avoid):
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
        return self._get_state(avoid)

    def step(self, action,avoid):
        done = False

        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation

        dis_target = simReadDistance(self.disT)
        # Reward is negative distance to target
        reward = -dis_target
        reward1 = reward

        dis= simReadDistance(self.dis0)

        c0 = simReadCollision(self.col)
        cG = simReadCollision(self.colG)
        if (c0):
            collision_robot = 1
        else:
            collision_robot = 0

        if (cG):
            collision_ground = 1
        else:
            collision_ground = 0

        if dis < thresh:
            reward = reward - penalty

        if (reward1 > -0.05 and collision_robot == 0 ):
            done=True
            reward= reward + penalty

        return reward, self._get_state(avoid), done, collision_robot

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def calculateDistance(self):
        bx, by, bz = self.obstacle.get_position()
        a1x,a1y, a1z = simGetObjectPosition(self.joint7, -1)
        a2x, a2y, a2z = simGetObjectPosition(self.joint6, -1)
        a3x, a3y, a3z = simGetObjectPosition(self.joint5, -1)
        a4x, a4y, a4z = simGetObjectPosition(self.joint4, -1)
        a5x, a5y, a5z = simGetObjectPosition(self.joint3, -1)
        a6x, a6y, a6z = simGetObjectPosition(self.joint2, -1)
        a7x, a7y, a7z = simGetObjectPosition(self.joint1, -1)
        a8x, a8y, a8z = self.agent_ee_tip.get_position()

        a1 = np.array((a1x, a1x, a1z))
        a2 = np.array((a2x, a2y, a2z))
        a3 = np.array((a3x, a3y, a3z))
        a4 = np.array((a4x, a4y, a4z))
        a5 = np.array((a5x, a5y, a5z))
        a6 = np.array((a6x, a6y, a6z))
        a7 = np.array((a7x, a7y, a7z))
        a8 = np.array((a8x, a8y, a8z))

        b = np.array((bx, by, bz))


        return min([np.linalg.norm(a1- b) , np.linalg.norm(a2- b) , np.linalg.norm(a3- b) , np.linalg.norm(a4- b) , np.linalg.norm(a5- b) ,
                        np.linalg.norm(a6 - b), np.linalg.norm(a7- b) , np.linalg.norm(a8- b) ])

# set env
env = ArmEnv()
s_dim = 20
a_dim = 7
a_bound = action_bound

# set RL method (continuous)
name2 = 'avoid'
name = 'reach'


avoid_graph = tf.Graph()
with avoid_graph.as_default():
    rl_avoid = DDPG(a_dim, s_dim, a_bound,ON_TRAIN, name2)
    print("rl avoid created........................................................")
    rl_avoid.restore(avoid_dir)
    print("rl avoid restored...........")
s_dim = 17
reach_graph = tf.Graph()
with reach_graph.as_default():
    rl_reach = DDPG(a_dim, s_dim, a_bound,ON_TRAIN, name)
    print("rl reach created..................................................")
    rl_reach.restore(reach_dir)
    print("rl reach restored.................................................")



def eval():
    reward_list = []
    x_axis = np.array(range(evalnum))

    col_list = np.zeros((evalnum), dtype=bool)
    done_list = np.zeros((evalnum), dtype=bool)
    for i in range(evalnum):
        avoid = False
        s = env.reset(avoid)
        ep_r = 0.
        done_count = 0

        for j in range(MAX_EP_STEPS):

            #dis = simReadDistance(env.dis0)   #calculate distance from simulation
            dis = env.calculateDistance()    #calculate distance similar to the real world
            if dis < thresh:
                avoid = True
                s = env._get_state(avoid)
                a = rl_avoid.choose_action(s)
            else:
                avoid = False
                s = env._get_state(avoid)
                a = rl_reach.choose_action(s)

            r, s_, done, col = env.step(a,avoid)

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
                if done_count==30 and col_list[i] == False:
                    done_list[i] = True
                break


    reward_list= np.array(reward_list)

    plt.plot(x_axis,reward_list,color='b')
    plt.plot(x_axis[col_list],reward_list[col_list], 'rx')
    plt.plot(x_axis[done_list],reward_list[done_list], 'go')
    plt.title('Tested on ' + str(evalnum)+ ' episodes with threshold '+ str(thresh)+'\n' +'Success rate: '+ str(100*len(reward_list[done_list]) / evalnum)+ '%' + 'Collusion rate: '+str(100*len(reward_list[col_list]) / evalnum )+'%')
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.savefig(dir+'testDynamicHybridthresh'+str(thresh)+'.png')


if ON_TRAIN:
    train()
else:
    eval()

print('Evaluation Done!')

env.shutdown()
