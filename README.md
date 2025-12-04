# AFO Project

This workflow filters and compute data from OSIM, MOT and TRC files to help the study of gait analysis.


## WIP pipeline

The pipeline in progress is located in [this](TreadMetrix/wip_pipeline) directory.

The pipeline in progress is [`full_pipeline.py`](TreadMetrix/wip_pipeline/full_pipeline.py).  
It uses custom MOT and TRC classes of the [`resources/filetypes_gestion`](resources/file_types) directory, 
and automatically follows the steps of the pipeline 
using functions located in [`data_postprocessing.py`](TreadMetrix/wip_pipeline/data_postprocessing.py).  

You just have to run [`full_pipeline.py`](TreadMetrix/wip_pipeline/full_pipeline.py).  


## Old pipeline  

The remaining files of the old pipeline are located in [this](TreadMetrix/old_pipeline) directory.
They are programmed for deletion once adapted into the new pipeline and may not function as intended in the meantime.


## Path management  

If existing, have a look at the [`local.json`](resources/paths/local.json) file.

> If it does not exist yet, don't worry: running the code will automatically create it.

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