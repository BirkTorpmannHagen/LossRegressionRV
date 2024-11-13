# Loss Regression
Repository for [paper link] 
Note for reviewers: refactoring in progress. Some things might not work quite yet.

## Reproducing results
To reproduce the results, it is necessary create a new conda environment using the environment.yaml file.
The data collection and analysis pipeline is as follows:
 - Train appropriate model(s), and specify their checkpoints in the corresponding TestBed class(es). 
 - Compute feature-loss pairs for a given testbed using ```eval_detectors.py```
 - Extract batched data statistics through bootstrapping, using ```simulate.py```
 - Read results and generate plots in ```plots.py```

### Training new networks
New networks can be trained by going into the approprite model folder (segmentation/classifier/glow), editing ```train.py``` for the desired dataset, and running it. 
For the resulting checkpoints to be loaded, specify the path to the checkpoint in the corresponding ```TestBed``` class in ```testbeds.py``` 

### Computing feature-loss pairs
Once a checkpoint is available and specified in the corresponding TestBed, featute-loss pairs can be extracted by executing ```eval_detectors.py```. This generates ```.csv``` files that contain feature-loss pairs organized according to the type of shift in ```single_data/```. 

### Training GAMs and extracting experimental data
Once there is data in ```single_data/```, run ```get_gam_data()``` to train GAMs. The resultng data is stored in ```gam_preds.csv``` and ```gam_results_reduction.csv``` 

