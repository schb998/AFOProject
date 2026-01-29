# AFO Project

This workflow filters and compute data from OSIM, C3D, MOT and TRC files to help the study of gait analysis.

## First steps & requirements

To use this pipeline, you should first clone this repository into your computer. 

Once this is done, ensure you ahe OpenSim installed, and run the [install requirements file](install_requirements.py).

This will ensure that you have a conda environment suitable to run the pipeline without issue.



## Pipeline

The files managing the pipeline are all located in the [TreadMetrix](TreadMetrix) directory. 

### In-line pipeline

TO run the in-line pipeline, look at the [`full_pipeline.py`](TreadMetrix/full_pipeline.py) file.  

It uses custom MOT and TRC classes of the [`resources/filetypes_gestion`](resources/file_types) directory, 
and automatically follows the steps of the pipeline using functions located in 
[`data_postprocessing.py`](TreadMetrix/data_postprocessing.py), [`ik_computing.py`](TreadMetrix/ik_computing.py),
[`id_computing.py`](TreadMetrix/id_computing.py) and [`joint_power_computing.py`](TreadMetrix/joint_power_computing.py).  

To run the pipeline, you just have to run [`full_pipeline.py`](TreadMetrix/full_pipeline.py).  

### GUI

To run the GUI version of the pipeline, head to the [`GUI.py`](TreadMetrix/GUI.py) file and run it.  

Using the previous pipeline, this GUI aims to give a more visual feedback to the pipeline process.



### Parameters selection

Upon running both versions of the pipeline, minimalists windows will pop up and ask for your inputs on the different 
parameters of the pipeline.

> The Quick Start Up option will run the pipeline with the paths of your save file (see Path management, below), and the 
minimalist parameters: saving as few files and showing as few plots as possible. the pipeline will crash if this option
is selected with no existing file save.

Those parameters are as follows:
- Whether to save optional files. Saving those files means that the pipeline will take the time and space to keep track of the files it doesn't necessarily need saved. This includes the corrected GRF data, the External Loads computes by opensim, as well as different plots.
- Whether to show the plots as the code run. That means interrupting the pipeline to give visual feedback to the user. The pipeline will consequently ask for more user input, as the shown plots need to be manually closed.
- Selecting the output directory and the files to process. Those are critical inputs for the pipeline. Please refer to the Inputs and Path management parts below.
- Selection of the OpenSim's bin directory. This is necessary for OpenSim to run the IK and ID tools.
- Selection of the scaled model. The scaled model is necessary for the accuracy of OpenSim's computation. There is the option to scale a new model.

The last three points are filled by default with the data from the save file, if existing.



## Inputs




## OpenSim tools

### Scaling

### IK & ID computing




## Path management  

If existing, have a look at the [`local.json`](resources/paths/local.json) file.

> If this file does not exist yet, don't worry: running the code will automatically create it.

This file will be used to locally store the paths to the files used in the code, such as the directory where modified 
files will be saved, or results of computations. It was preemptively added to the [`.gitignore`](.gitignore) file, which 
means it's not going to be pushed to the GitHub repository.

When running the code, a window will appear and ask you to select paths and files that will be used during the pipeline. 
This includes:  
- `output path`, a directory to save the new files in, see details [below](#output-structure).
- `raw MOTs`, the MOT files to process.  
- `raw TRCs`, the TRC files to process.  
- `scaled model`: if existing, you can select the scaled OSIM model of your participant.
  If it does not exist, it will be generated as the code run using the following files:  
  - `base model`: the OSIM model that will be scaled.  
  - `scaling setup`: the XML file that will be used to scale the base model.  

### Output structure

After running the code, the output directory will be as follows, some directories put aside 
depending on file saving preferences:  
```
└── output_path
    ├── corrected_mot
    ├── external_loads
    ├── id_results
    ├── ik_results
    ├── power_filtered
    └── segmented
        ├── mot
        └── trc
```