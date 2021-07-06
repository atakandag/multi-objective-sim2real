# multi-objective sim2real

## Paper
Monolithic vs. Hybrid Controller for Multi-objective Sim-to-Real Learning (A. Dag, A. Angleraud, W. Yang, N. Strokina, R.S. Pieters, M. Lanz and J.-K. Kämäräinen), In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021

## Video
Supplementary video can be found at https://www.youtube.com/watch?v=8t_0aoA6gOk

## Requirements
* CopelliaSim (formerly known as V-REP) (https://www.coppeliarobotics.com/)
* PyRep (https://github.com/stepjam/PyRep)
* TensorFlow 1.14

## Launching
You can directly run 'mono.py' and 'hybrid.py' files to test the behaviour. If you want to change the speed of the robot: change 
```action_bound = [-0.02, 0.02]``` in line 25.

If you want to train your own model by using 'mono.py', 'hybrid_reach.py' or 'hybrid.avoid.py':
* Make ```ON_TRAIN = True``` in line 21
* Change ```dir = './models/reach/params1/'``` in line 37 otherwise your parameters will be saved to an existing model.
* Remember to make ```action_bound = [-1, 1]``` in line 25 during the training to have better exploration by increasing the velocity limits.
* You can use your own scene by adding to 'scenes' folder and changing the ```SCENE_FILE = './scenes/scene_reinforcement_learning_env_5_RealObsDynRandom.ttt'``` in line 23. Remember to modify the environment class (ArmEnv) accordingly if you use your own scene.

## Notes
Only the best performing models are shared in this repository. 

DDPG implementation ('rl.py' file) is a modified version of https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG_update.py

If you already have TensorFlow 2, you can simply add:
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```
