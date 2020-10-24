# sim2real2020

## Requirements
* Copellia Sim (formerly known as V-REP) (https://www.coppeliarobotics.com/)
* PyRep (https://github.com/stepjam/PyRep)
* TensorFlow 1

## Launching
You can directly run 'mono.py' and 'hybrid.py' files to test the behaviour. 

If you want to train another model by using 'mono.py', 'hybrid_reach.py' or 'hybrid.avoid.py':
* Make 'ON_TRAIN = True'
* Crate a directory to hold the parameters and remember to change 'dir' in the code.
* You can use your own scene by adding to 'scenes' folder and changing the 'SCENE_FILE' in the code. Remember to modify the environment class accordingly if you use your own scene.

## Notes
Only the best performing models are shared in this repository. 
