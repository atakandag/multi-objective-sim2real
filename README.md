# sim2real2020

## Requirements
* CopelliaSim (formerly known as V-REP) (https://www.coppeliarobotics.com/)
* PyRep (https://github.com/stepjam/PyRep)
* TensorFlow 1.14

## Launching
You can directly run 'mono.py' and 'hybrid.py' files to test the behaviour. 

If you want to train another model by using 'mono.py', 'hybrid_reach.py' or 'hybrid.avoid.py':
* Make 'ON_TRAIN = True'
* Change "dir = './models/reach/params1/'" in line 37 otherwise your parameters will be saved to an existing model.
* You can use your own scene by adding to 'scenes' folder and changing the 'SCENE_FILE' in the code. Remember to modify the environment class accordingly if you use your own scene.

## Notes
Only the best performing models are shared in this repository. 

DDPG implementation ('rl.py' file) is a modified version of https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG_update.py
