# Sales Performance Classification
Interpretable classification model to identify high-performing sales representatives using demographic, experience, compensation, and business-unit features, with emphasis on preventing leakage and evaluating threshold tradeoffs.


## Highlights
- High performer label engineering using top-20% thresholds in NPS and Feedback
- Logistic regression baseline with class imbalance awareness
- Probability threshold tuning to quantify precision/recall tradeoffs across business use cases


## Tech Stack 
Python, pandas, NumPy, scikit-learn, matplotlib

## Key Takeaway 
Threshold-tuned classification enables flexible identification of high-performing sales representatives, capturing over 90% of top performers at lower thresholds while allowing precision to increase for high-confidence targeting. The results emphasize how probability-based models support differentstrategies depending on business objectives.
