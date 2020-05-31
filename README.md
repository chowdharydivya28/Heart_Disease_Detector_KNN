# Heart_Disease_Detector_KNN
Using KNN model to predict whether or not an individual has chances of developing heart disease based on features obtained from their medical records

# Model Evaluation

**Without Scaling**
* Best K = 11
* Best distance function = Gaussian kernel distance:
  d(x,y)=exp(−12⟨x−y,x−y⟩)
* F1-score = 0.83

**With Scaling**
* Best K = 5
* Best distance function = Euclidean distance:
  d(x,y)=sqrt(⟨x−y,x−y))
* Best scaler = min_max_scale
* F1-score = 0.89

