import streamlit as st
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from model import load_or_train_lstm, predict_future

st.set_page_config(layout='wide')
st.title('AAPL Stock Price Prediction with LSTM')

@st.cache_data
def load_data():
    df = pd.read_csv('AAPL.csv')
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Remove timezone info!
    df = df.sort_values('date')
    return df

df = load_data()

st.sidebar.header('Options')

# Date range
min_date = df['date'].min().date()
max_date = df['date'].max().date()
try:
    start_date, end_date = st.sidebar.date_input(
        "Date range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
except Exception:
    start_date, end_date = min_date, max_date

filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

if st.sidebar.checkbox('Show raw data'):
    st.write(filtered)

# LSTM parameters
lookback = st.sidebar.slider("Lookback window (days)", min_value=20, max_value=120, value=60)
pred_days = st.sidebar.slider("Predict next N days", min_value=1, max_value=30, value=7)

# Show chart
st.subheader('Historical Closing Prices')
st.line_chart(filtered.set_index('date')['close'])

if st.sidebar.button('Train model (if no model exists)'):
    with st.spinner('Training model (may take a minute)...'):
        _, _ = load_or_train_lstm(df, lookback)
    st.success("Model trained and saved!")

if st.sidebar.button('Predict Future Prices'):
    with st.spinner('Loading/training model...'):
        model, scaler = load_or_train_lstm(df, lookback)
    with st.spinner('Predicting future prices...'):
        preds = predict_future(df, lookback, pred_days, model, scaler)
    last_date = df['date'].max()
    fut_dates = [last_date + timedelta(days=i+1) for i in range(pred_days)]
    pred_df = pd.DataFrame({'date': fut_dates, 'predicted_close': preds})
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(df['date'], df['close'], label='Historical')
    plt.plot(pred_df['date'], pred_df['predicted_close'], label='Predicted', marker='o')
    plt.legend()
    plt.xlabel('Date'); plt.ylabel('Close Price')
    plt.title('AAPL Close Price Prediction')
    st.pyplot(plt)
    st.write(pred_df)
    st.download_button("Download Predictions", pred_df.to_csv(index=False), file_name='predictions.csv', mime='text/csv')

st.markdown("""
---
* Place your `AAPL.csv` file in this folder.
* Use sidebar buttons to train (first run) or predict.
* Adjust lookback/prediction window as desired.
* Download results after prediction.
""")
