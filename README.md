# AFOProject
This workflow filters and compute data from OSIM, MOT and TRC files to help the study of grait analysis using OpenSim and a treadmill.

## Pipeline
1. [`Treadmetrix.py`](TreadMetrix/TreadMetrix.py) filters a MOT file using the Butterworth filter.
2. [`mot.cut.steps.py`](TreadMetrix/mot.cut.steps.py) segments a filtered MOT file into multiple MOT files, separated by gait cycle (heelstrike to following heelstrike) and leg side.
3. [`trc.cut.steps.py`](TreadMetrix/trc.cut.steps.py) segments a TRC file by side and gait cycle, from the corresponding filtered MOT file.
4. [`IK_pipeline_mot`](TreadMetrix/IK_pipeline_mot.py) computes Inverse Kinematics from the data.
5. [`RunID_Treadmill`](TreadMetrix/RunID_Treadmill.py) computes Inverse Dynamics from the data.
6. [`Joint_power_mot_Treadmill`](TreadMetrix/Joint_power_mot_Treadmill.py) computes the joint power from the Ik and ID data.

## How to use it?
First thing first, you'll need to have a look at the [`local.py`](TreadMetrix/local.py) file. 
This file is there to adapt to the computer and project specificities. 
Fill the missing data as indicated. 
The [`TreadMetrix/local_paths.py`](TreadMetrix/local_paths.py) will then be used to transfer that data and standardize the project structure.  

Once this is done, uncomment the indicated line in your .gitignore file.

Then, you just have to run the files of the pipeline in the order above. 