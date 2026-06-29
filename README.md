# Financial-Management-App
# Stock Price Forecasting — Multi-Model Comparison

Predicts stock prices for multiple companies using LSTM, GRU, 
Attention-LSTM, and ARIMA. Models are trained and evaluated 
side-by-side to identify which generalizes best.



## Models & Results

Average performance across all companies:

| Model          | MSE      | RMSE    | MAE     | R²     |
|----------------|----------|---------|---------|--------|
| LSTM           | 189.1159 | 10.3343 | 8.1521  | 0.8984 |
| GRU            | 49.8607  | 5.6275  | 4.2937  | 0.9677 |
| Attention-LSTM | 625.6468 | 17.7002 | 13.9670 | 0.7147 |

### Model Ranking

| Rank | Model          | Remarks |
|------|----------------|---------|
|  1 | GRU            | Best predictive performance across all metrics |
|  2 | LSTM           | Good accuracy with stable predictions |
|  3 | Attention-LSTM | Higher error and lower explanatory power |

## Key Findings

Among all evaluated deep learning architectures, the **GRU model consistently outperformed the others**, achieving the lowest forecasting errors and the highest coefficient of determination (R² = 0.9677). This indicates that GRU was most effective at capturing temporal patterns in the stock price data across all companies considered in this study.

## Tech Stack
Python, TensorFlow/Keras, Scikit-learn, Statsmodels, 
pmdarima, Pandas, Matplotlib, Streamlit

## How to Run
# Install dependencies pip install -r requirements.txt # Train all models and save .npy outputs python src/train.py # Generate evaluation metrics and plots python src/evaluate.py # Launch Streamlit dashboard streamlit run app.py

## Results 
![Combined predictions](assets/combined_predictions.png)
