# Import necessary libraries
import os
import streamlit as st
import pandas as pd
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import hvplot.pandas

# Define the Streamlit app title
st.title("US Stock Market Analysis")

# Function to fetch historical stock data
def fetch_stock_data(assets, start_date, end_date):
    assets_data = {}
    for asset in assets:
        data = si.get_data(asset, start_date=start_date, end_date=end_date)
        data = data.reset_index()
        data = data.drop(columns=['ticker'], axis=1)
        data = data.rename(columns={'index': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'adjclose': 'Adj Close', 'volume': 'Volume'})
        data = data.set_index('Date')
        assets_data[asset] = data
    return assets_data

def visualize_data(data, chart_type='line'):
    if chart_type == 'line':
        for asset, df in data.items():
            st.title(f'{asset} Historical Closing Prices')
            st.line_chart(df['Close'])
    elif chart_type == 'bar':
        for asset, df in data.items():
            st.title(f'{asset} Volume')
            st.bar_chart(df['Volume'])

# Define fetch_stock_data and visualize_data functions as shown in the previous response

# Define assets_data in the global scope
assets_data = None

# Sidebar inputs for date range and assets
start_date = st.sidebar.date_input("Select the start date:", value=datetime.datetime.now() - datetime.timedelta(days=365 * 10))
end_date = st.sidebar.date_input("Select the end date:", value=datetime.datetime.now())
assets = st.sidebar.multiselect("Select assets:", ["^GSPC", "AAPL", "ABBV", "AMZN", "BAC", "CVX", "GOOG", "JNJ", "KO", "MA", "MSFT", "NVDA", "PEP", "PG", "TSLA", "V", "BRK-B", "BTC-USD", "ETH-USD", "PFE", "UNH", "XOM", "^TNX", "^TYX"])

# Fetch data for selected assets
if st.sidebar.button("Fetch Data"):
    st.info("Fetching data... This might take a moment.")
    assets_data = fetch_stock_data(assets, start_date=start_date, end_date=end_date)
    st.success("Data fetched successfully!")

# Additional visualizations and analysis
if st.sidebar.button("Show Analysis"):
    st.info("Fetching data... This might take a moment.")
    assets_data = fetch_stock_data(assets, start_date=start_date, end_date=end_date)
    if assets_data:
        st.success("Data fetched successfully!")

        
        st.write("### Adjusted Closing Prices")
        for asset, data in assets_data.items():
            st.title(f'{asset} Adjusted Closing Prices')
            st.line_chart(data['Adj Close'])

        st.write("### Cumulative Returns")
        cumulative_returns = pd.DataFrame({asset: data['Adj Close'].pct_change().cumsum() for asset, data in assets_data.items()})
        st.line_chart(cumulative_returns)

        st.write("### Sharpe Ratios")
        returns = pd.DataFrame({asset: data['Adj Close'].pct_change() for asset, data in assets_data.items()})
        risk_free_rate = st.sidebar.number_input("Enter the risk-free rate:", value=0.03, step=0.01)
        sharpe_ratios = ((returns.mean() - risk_free_rate) / returns.std()).sort_values(ascending=False)
        st.bar_chart(sharpe_ratios)

        st.write("### Rolling Mean of Asset Prices")
        window = st.sidebar.slider("Select window size:", min_value=5, max_value=100, value=50, step=5)
        rolling_mean = pd.DataFrame({asset: data['Adj Close'].rolling(window=window).mean() for asset, data in assets_data.items()})
        st.line_chart(rolling_mean)

        st.write("### Historical Volatility of Assets")
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualize the volatility
        st.line_chart(volatility)

        st.write("### Correlation Matrix of Adjusted Closing Prices")
        correlation_matrix = returns.corr()
        st.write(correlation_matrix)

        st.write("### Residual Plots")
        for asset, data in assets_data.items():
            features = data['Adj Close'].shift(1).dropna()
            target = data['Adj Close'].dropna()
            # Align the lengths of features and target arrays
            min_length = min(len(features), len(target))
            features = features[:min_length]
            target = target[:min_length]
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train.values.reshape(-1, 1), y_train)
            predictions = model.predict(X_test.values.reshape(-1, 1))
            residuals = y_test - predictions
            fig, ax = plt.subplots()
            ax.scatter(X_test, residuals, label=f'{asset} Residuals')
            ax.axhline(y=0, color='black', linestyle='--', label='Zero Residuals')
            ax.set_xlabel(f'Previous day\'s Adj Close - {asset}')
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residual Plot for {asset} Stock')
            ax.legend()
            st.pyplot(fig)
    else:
        st.error("Failed to fetch data. Please check your inputs and try again.")

# Define a function to download data as CSV
def download_csv(data):
    csv_data = pd.concat(data.values(), keys=data.keys())
    csv_data.to_csv("stock_data.csv", index=False)
    st.markdown("### [Download CSV](stock_data.csv)")

# Define a function to download data as JSON
def download_json(data):
    json_data = {asset: df.to_dict(orient='records') for asset, df in data.items()}
    with open("stock_data.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    st.markdown("### [Download JSON](stock_data.json)")

# Define a function to download data as PDF
def download_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="US Stock Market Analysis", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt="Data Summary", ln=True, align="C")
    pdf.ln(10)
    for asset, df in assets_data.items():
        pdf.cell(200, 10, txt=f"{asset}:", ln=True, align="L")
        pdf.cell(200, 10, txt=df.describe().to_string(), ln=True, align="L")
        pdf.ln(5)
    pdf.output("stock_data_summary.pdf")
    st.markdown("### [Download PDF](stock_data_summary.pdf)")

# Show download options
if assets_data:
    st.write("### Download Data")
    if st.button("Download CSV"):
        download_csv(assets_data)
    if st.button("Download JSON"):
        download_json(assets_data)
    if st.button("Download PDF"):
        download_pdf()