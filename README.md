# AFO Project

This workflow filters and compute data from OSIM, MOT and TRC files to help the study of gait analysis.  

## Path management

Have a look at the [`local_paths.py`](TreadMetrix/local_paths.py) file. This file is there to adapt to the computer and project specificities. 
It uses a file called `.local.json` that is currently missing.  

`.local.json` needs to be created and added to the repository. 
It will contain paths that are local to the computer and its directories
and will not be pushed to the GitHub repository.  

### Update paths

Manually add the `.local.json` file into the [`TreadMetrix directory`](\TreadMetrix).  

Then copy-paste and fill the following lines into `.local.json`:  
```json
{
  "opensim_path":      "fill in with absolute path to bin file",
  "base_path":         "fill in with absolute path to directory",
  "osim_scaled_model": "fill in with absolute path to file"
}
```

> **To be noted:** some of those paths will require to be updated for each participant.

Additional paths can be added: `"raw_mot"` & `"raw_trc"`, `"corrected_mot"`, `"segmented_mot"` & 
`"segmented_trc"`, `"external_loads"`, `"ik_results"` & `"id_results"`, `"power_filtered"`.

If those optional paths are not defined, the base_path directory will be assumed to be as follows:  

```
└── base_path
    └── raw
        ├── raw_mot.mot
        └── raw_trc.trc
```

> **To be noted:** the `raw` directory illustrated can contain multiple MOT and TRC files. 
> They will all be processed by the pipeline. 
>> The WIP pipeline do not process files containing the term "static".

And the directory will be as follows after going through the pipeline (some directories put aside, 
depending on file saving preferences):  
```
└── base_path
    ├── corrected_mot
    ├── external_loads
    ├── id_results
    ├── ik_results
    ├── power_filtered
    ├── raw
    └── segmented
        ├── mot
        └── trc
```

## Current pipeline

1. [`Treadmetrix.py`](TreadMetrix/TreadMetrix.py) filters a MOT file using the Butterworth filter.  
2. [`mot.cut.steps.py`](TreadMetrix/mot.cut.steps.py) segments a filtered MOT file into multiple MOT files, 
separated by gait cycle (heelstrike to following heelstrike) and leg side.  
3. [`trc.cut.steps.py`](TreadMetrix/trc.cut.steps.py) segments a TRC file by side and gait cycle, 
from the corresponding filtered MOT file.  
4. [`IK_pipeline_mot`](TreadMetrix/IK_pipeline_mot.py) computes Inverse Kinematics from the data.  
5. [`RunID_Treadmill`](TreadMetrix/RunID_Treadmill.py) computes Inverse Dynamics from the data.  
6. [`Joint_power_mot_Treadmill`](TreadMetrix/Joint_power_mot_Treadmill.py) computes the joint power from the Ik and ID data.  

Once [the local paths have been filled](#update-paths), run the files of the pipeline in the order above.  


## WIP pipeline

The pipeline in progress is [`full_pipeline.py`](TreadMetrix/full_pipeline.py).  
It uses custom MOT and TRC classes of the [`resources/filetypes_gestion`](resources/filetypes_gestion) directory, 
and automatically follows the steps of the pipeline 
using functions located in [`data_postprocessing.py`](TreadMetrix/data_postprocessing.py).  

Once [the local paths have been filled](#update-paths), run the pipeline file.