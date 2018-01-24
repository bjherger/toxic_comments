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
   
  