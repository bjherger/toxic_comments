# Toxic Comments

**tl;dr:** Surfacing toxic Wikipedia comments, by training an NLP deep learning model utilizing multi-task learning a 
variety of deep learning architectures. Data from a 
[Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Quick start

```bash
# Install anaconda environment
conda env create -f environment.yml 

# Activate environment
source activate toxic

# Run script
cd bin/
python main.py
```

## Repo structure

 - `bin/main.py`: Code entry point
 - `docs/writeup/writeup.md`: Project summary
 - `conf/confs.yaml`: Configuration file, used to choose parameters
 - `docs/modeling_notes.md`: Notes / support for design decisions
 - `data/schemas`: Data set schemas

## Python Environment
Python code in this repo utilizes packages that are not part of the common library. To make sure you have all of the 
appropriate packages, please install [Anaconda](https://www.continuum.io/downloads), and install the environment 
described in environment.yml (Instructions [here](http://conda.pydata.org/docs/using/envs.html)). 

## Configuration file

This program utilizes a configuration file (`conf/confs.yaml`). It will run with the default parameters, but many 
parameters can be freely changed. Parameters include:

 - `run_train`: Whether to train a model
 - `run_infer`: Whether to used the trained model to predict classifications for the Kaggle test data set
 - `test_run`: Whether to run on a subset of observations. This is helpful for debugging
 - `model_choice`: Either the name of a method in `bin/models`, or serialized. If serialized, use 
 `serialized_model_choice_path` to provide a path to a serialized model
 - `num_epochs`: The number of epochs to train for

## Contact
Feel free to contact me at 13herger `<at>` gmail `<dot>` com
