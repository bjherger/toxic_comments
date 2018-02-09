# Writeup Outline

## Outline

Intro

 - Goal: Compare character level model architectures, play w/ multi-class outputs
 - Kaggle: About cheap, labeled data. Not about competition

Data

 - Point to Kaggle's About Us
 - Unique combination of text, mutliple outputs
 - Character vs Token
 
Model
 - Section on metrics
   - Difficulty of rare-event modeling
   - Avg log loss
   - AUC
 - Discuss losses
 - Discuss Keras's labels are mutually exclusive assumption for log-loss
 - Multiple models, transfer learning / shared weights, multi-output model
   - Training, backprop, deployment / state, loss optimization

Conclusion

 - Results
 - Future work
   - Token level models
   - Wider networks (more LSTM layers)
   - Optimize learning rate further
   - Coarse / fine filter (is toxic or not, which kind of toxic)