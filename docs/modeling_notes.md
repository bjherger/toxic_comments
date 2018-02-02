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
 