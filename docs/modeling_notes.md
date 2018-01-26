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
 - Implement submission formatter
 - Rename repo
 - Copy over character model from spoilers
 - Train first pass at model

Metric

 - Keras wants [column wise log-loss](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#evaluation)
 - log-loss is the default accuracy metric w/ binary models , according to this [so](https://stackoverflow.com/a/37156047)
 - Should be able to use [categorical cross-entropy](https://keras.io/losses/#categorical_crossentropy), and report 
 back `accuracy`
 
 