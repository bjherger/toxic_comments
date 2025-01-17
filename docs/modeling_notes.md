# Modeling Notes

## 2018-01-23

Backlog

 - Evaluate histograms
   - Num words per post, num characters per post, unique words in post / post length
 - Stats
   - num unique words (vocab size), % of posts with at least one flag
 - Models
   - Character / token models
   - Coarse (toxic or not) / fine model (kind of toxic)
   - Straight multi-output model
   
Prioritized backlog

 - Histograms
 - Stats
 - Determine model architecture
 
### Extract

 - No notes
 
### Transform

 - pandas max, should use `axis=1` for max across multiple rows
 - Some toxic types are incredibly rare. It might be worth always setting their probability to zero 

## 2018-01-24

### Transform

 - Writing histogram helper method
 - Creating histograms
 
## 2018-01-26

### Character model

TODO

 - Implement metric
 - Implement baseline model
 - Implement callbacks
 - Implement submission formatter
 
 - Rename repo
 - Copy over character model from spoilers
 - Train first pass at model

Metric

 - Keras wants [column wise log-loss](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#evaluation)
 - log-loss is the default accuracy metric w/ binary models , according to this [so](https://stackoverflow.com/a/37156047)
 - Should be able to use [categorical cross-entropy](https://keras.io/losses/#categorical_crossentropy), and report 
 back `accuracy`
 
Baseline model

 - Stupid, minimal model

Callbacks

 - Borrowing from [spoilers](https://github.com/bjherger/spoilers_model/blob/master/bin/main.py)
 - They seem to work
 
Refactor

 - Refactoring into train / infer steps
 - Updating load code
 

 
## 2018-01-27

Submission

 - Formatting output for submission
 - Testing stupid model submission
 - Reviewing [unofficial kaggle cli](https://github.com/floydwch/kaggle-cli)
 - Submitting baseline gaussian random input data set
 - Submitting baseline zeros input data set

 - Validation loss is much lower than submission loss
 
Metric

 - Loss should be consistent w/ columnwise log loss. Looking into it a bit more.
 

 
 ## 2018-01-28
 
Response Var

 - Setting up response var as separate Ys
 - Setting up mutliple y arrays in transform
 - Setting up multiple output layers
 - Intalling TF and h5py (not in environment)


## 2018-02-06

Bince ast update:

 - Set up train / infer pipeline
 - Set up biLSTM model
 - Trained a few simple models
 
Refactor. A few goals:

 - Put all run output in a single folder (easier archiving)
 - README
 - Easier inference from serialized model
 - Continue training serialized model
 - Easier ability to switch to different data sets
 - Easier to follow logic around X and Y transformation
 - De-couple logic for character level mode from this particular data set
 - Specify model in conf (?)
 - Create model architecture in conf (?)
 - Docstrings
 - Model run summary
 - Add AUC metric
 
Better models. A few ideas:

 - Convolution layer (reduce train time)
 - Vary learning rate
 - Vary embedding size 
 - Dense layer after bi-lstm
 
Prioritized backlog:

 - Project wide code style refactor
 - De-couple logic for character level mode from this particular data set
   - Easier to follow logic around X and Y transformation
   - Put all run output in a single folder (easier archiving)
   - Docstrings
 - README
 - Model integration
   - Specify model in conf (?)
   - Easier inference from serialized model
   - Continue training serialized model
   - Model run summary
 - Add AUC metric
 
Project wide code style refactor

 - Pycharm auto-refactor
 - Pycharm inspect code
 - There are a few places w/ capitalized variables (`X`), and other misc issues that I choose to leave in

### De-couple logic

TODO

 - Create model conf parameter
 - Create model singleton
 - Modify load to create and fill a batch file
 - More consistent logging
 - Do something else w/ EDA code
 - Add flag for train
 - Add flag for inference
 - Clean up confs file / Add comments
 
TODONE

 - Create model conf parameter
 - Create model singleton
 - Modify load to create and fill a batch file
   - Include model type in batch name
   - ISO 8601 formatted time for batchname
   - Model run summary
 - More consistent logging
 - Do something else w/ EDA code (Will not do)
 - Add flag for train
 - Add flag for inference
 - Clean up confs file / Add comments
 
### Model

 - Adding conv layer
 - Adding conv layer w/ stride
 - Adding Dense layer at end w/ linear
 - Adding Dense layer at end w/ relu
 
### TODO

 - Add \r and \n back into set of legal characters
 - README
 - AUC Metric
 - Model run summary file
 - Docstrings
 - Try smaller embedding size
 - Try different learning rate
 
## 2018-02-09

### Backlog

TODO

 - Update models to conv layers compile
 - Outline writeup
 
### Conv layers

Filter parameter

 - Search results aren't helpful
 - Looking through sourcecode to see where param is used
 - Changed parameter, but don't see a change in output tensors on tensorboard
 - Change seems to be in convolution kernel
 - Found a [so](https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d)
 - Basically a different set of learned weights. filters=1 would just be one learned set of weights
 
### Writeup

Writing outline

### LSTM

 - Trying CuDNNLSTM
 - Too much work. Will do later
 
### TODO

 - CuDNNLSTM
 - Confirm Ys are boolean
 - Confirm X datatype

## 2018-02-13

### Backlog

 - Second pass at writeup
 - README
 - Highlight best model
 - Actual model metrics (?) 