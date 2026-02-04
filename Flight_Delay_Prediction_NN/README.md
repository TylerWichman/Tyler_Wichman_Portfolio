# Pre-Departure Flight Delay Prediction (Neural Network)

End-to-end machine learning pipeline that predicts whether a U.S. commercial flight will arrive 15+ minutes late using only pre-departure and schedule-known information, enabling proactive delay risk screening.

## Highlights
- Large-scale data cleaning and validation on ~2 million flight records  
- Careful handling of structural missingness and class imbalance  
- Leakage-safe feature selection restricted to pre-departure variables  
- Baseline logistic regression compared against a neural network model  
- Threshold-based evaluation emphasizing operational decision tradeoffs  

## Tech Stack
Python, pandas, NumPy, scikit-learn, TensorFlow, Keras   

## Key Takeaway
The neural network provides a modest improvement in overall ranking performance over a logistic baseline, while delivering a substantial operational benefit by identifying over 80% of late-arriving flights at selected thresholds. The project highlights the importance of model evaluation, threshold selection, and business context when deploying machine learning models in real-world operational settings.
