# Gym Member Churn Prediction with Weather Features

End-to-end churn modeling pipeline integrating gym membership data with external weather features to evaluate potential environmental effects on churn behavior.

## Highlights
- Data ingestion and validation in BigQuery
- External API integration and monthly feature aggregation
- Logistic regression modeling using BigQuery ML
- Emphasis on reliability, interpretability, and reproducibility

## Tech Stack 
Python, SQL, BigQuery, BigQuery ML

## Key Takeaway
The churn model demonstrates that incorporating external weather data provides a modest but measurable signal beyond core member attributes, with temperature and wind conditions contributing to churn risk in interpretable ways. The analysis highlights the importance of threshold selection and model interpretability when using churn predictions.
