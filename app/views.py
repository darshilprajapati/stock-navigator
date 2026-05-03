from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext
import requests
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

import datetime as dt
# import qrcode

from .models import Project

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm

import requests
import pandas as pd

from .utils.stock_data_provider import dummy_provider

API_KEY = "jbM7zJnUVCJcCcN3pT6OOnThw8SOWpak"  # Polygon.io API key

company = ['AAPL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
companyToSearch = ['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM', 'TSLA', 'MSFT', 'GOOGL', 'FB',]
def fetch_stock_data(symbol, start_date="2024-04-01", end_date="2025-03-04"):
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={API_KEY}"
        print(f"Fetching data for {symbol} from API...")
        response = requests.get(url)
        data = response.json()
        print(f"API Response for {symbol}:", data.get('status'))  # Print API status

        if "results" in data:
            print(f"Successfully got data for {symbol}")
            df = pd.DataFrame(data["results"])
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df.set_index("date", inplace=True)
            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
            return df
        else:
            print(f"No results in API response for {symbol}, using dummy data")
            return dummy_provider.get_dummy_data(symbol, start_date, end_date)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return dummy_provider.get_dummy_data(symbol, start_date, end_date)

def validate_ticker(ticker):
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    return "results" in data and len(data["results"]) > 0


def get_valid_tickers():
    tickers = ['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM']
    return [t for t in tickers if validate_ticker(t)]
    
# The Home page when Server loads up

def index(request):
    tickers = company
    stock_data = {}
    recent_stocks = []
    
    # Fetch data for each stock
    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        df = fetch_stock_data(ticker)
        if df is not None:
            stock_data[ticker] = df
            # Get the most recent data point and format it
            latest_data = {
                'ticker': ticker,  # lowercase to match template
                'open': round(float(df['Open'].iloc[-1]), 2),
                'high': round(float(df['High'].iloc[-1]), 2),
                'low': round(float(df['Low'].iloc[-1]), 2),
                'close': round(float(df['Close'].iloc[-1]), 2),
                'volume': int(df['Volume'].iloc[-1])
            }
            recent_stocks.append(latest_data)
            print(f"Added data for {ticker}: {latest_data}")

    # Create Plotly Graph
    fig_left = go.Figure()
    for ticker, df in stock_data.items():
        fig_left.add_trace(go.Scatter(x=df.index, y=df["Close"], name=ticker))

    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_left = plot(fig_left, auto_open=False, output_type='div')

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })


def search(request):
    return render(request, 'search.html', {})

def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })


# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        print(f"\n=== Starting prediction for {ticker_value} for next {number_of_days} days ===")
        
        # Validate inputs
        ticker_value = ticker_value.upper()
        number_of_days = int(number_of_days)
        
        # Validate ticker and days
        if ticker_value not in companyToSearch:
            print(f"Invalid ticker: {ticker_value}")
            return render(request, 'Invalid_Ticker.html', {})
        if number_of_days < 0:
            print(f"Invalid days: {number_of_days}")
            return render(request, 'Negative_Days.html', {})
        if number_of_days > 365:
            print(f"Days overflow: {number_of_days}")
            return render(request, 'Overflow_days.html', {})

        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        future_dates = pd.date_range(
            start=end_date,
            periods=number_of_days + 1,
            freq='B'
        )[1:]

        print(f"Historical data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Predicting for dates: {future_dates[0].strftime('%Y-%m-%d')} to {future_dates[-1].strftime('%Y-%m-%d')}")

        # Fetch historical data
        df = fetch_stock_data(ticker_value, 
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'))
        
        if df is None or df.empty:
            print("No data received from API")
            return render(request, 'API_Down.html', {})

        # Create candlestick chart for historical data
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='market data'
        ))
        fig.update_layout(
            title=f'{ticker_value} Stock Price Evolution',
            yaxis_title='Stock Price (USD)',
            paper_bgcolor="#14151b",
            plot_bgcolor="#14151b",
            font_color="white"
        )
        plot_div = plot(fig, auto_open=False, output_type='div')

        # Prepare prediction features with more indicators
        features = ['Close', 'MA5', 'MA20', 'Daily_Return', 'Volatility', 'Price_Momentum', 'Volume_Change']
        
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        df['Price_Momentum'] = df['Close'] - df['Close'].shift(5)
        df['Volume_Change'] = df['Volume'].pct_change()
        df = df.dropna()

        # Create feature matrix with proper column names
        X = pd.DataFrame(df[features])
        y = df['Close']
        
        # Scale features while preserving column names
        scaler = preprocessing.StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=features,
            index=X.index
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Generate predictions with dynamic feature updates
        predictions = []
        dates = []
        last_known_price = df['Close'].iloc[-1]
        last_known_volume = df['Volume'].iloc[-1]
        
        # Create initial feature window as DataFrame
        feature_window = pd.DataFrame(columns=features)
        feature_window['Close'] = df['Close'].tail(20)
        feature_window['MA5'] = df['MA5'].tail(20)
        feature_window['MA20'] = df['MA20'].tail(20)
        feature_window['Daily_Return'] = df['Daily_Return'].tail(20)
        feature_window['Volatility'] = df['Volatility'].tail(20)
        feature_window['Price_Momentum'] = df['Price_Momentum'].tail(20)
        feature_window['Volume_Change'] = df['Volume_Change'].tail(20)

        for i in range(number_of_days):
            # Create feature vector as DataFrame
            current_features = pd.DataFrame([feature_window.iloc[-1]], columns=features)
            
            # Make prediction
            scaled_features = pd.DataFrame(
                scaler.transform(current_features),
                columns=features
            )
            pred = model.predict(scaled_features)[0]
            
            # Add some randomness based on historical volatility
            volatility = feature_window['Volatility'].iloc[-1]
            volatility_factor = volatility * np.random.normal(0, 0.5)
            pred = pred * (1 + volatility_factor)
            
            # Store prediction
            predictions.append(pred)
            dates.append(future_dates[i])
            
            # Update feature window
            new_row = pd.Series({
                'Close': pred,
                'MA5': feature_window['Close'].tail(5).mean(),
                'MA20': feature_window['Close'].tail(20).mean(),
                'Daily_Return': (pred - feature_window['Close'].iloc[-1]) / feature_window['Close'].iloc[-1],
                'Volatility': feature_window['Daily_Return'].std(),
                'Price_Momentum': pred - feature_window['Close'].iloc[-5],
                'Volume_Change': np.random.normal(0, 0.1)
            })
            
            feature_window = pd.concat([feature_window[1:], pd.DataFrame([new_row])], ignore_index=True)

        # Create prediction plot with enhanced visualization
        pred_fig = go.Figure()
        
        # Add historical data
        pred_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions
        pred_fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            name='Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        # Add confidence interval
        std_dev = df['Close'].std()
        pred_fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=[p + std_dev for p in predictions] + [p - std_dev for p in predictions][::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Confidence Interval'
        ))

        pred_fig.update_layout(
            title=f'{ticker_value} Price Prediction',
            paper_bgcolor="#14151b",
            plot_bgcolor="#14151b",
            font_color="white"
        )
        plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

        # Prepare context
        context = {
            'plot_div': plot_div,
            'plot_div_pred': plot_div_pred,
            'ticker_value': ticker_value,
            'number_of_days': number_of_days,
            'current_price': f"${df['Close'].iloc[-1]:.2f}",
            'next_day_price': f"${predictions[0]:.2f}",
            'second_day_price': f"${predictions[1]:.2f}" if len(predictions) > 1 else "N/A",
            'next_day_date': dates[0].strftime('%Y-%m-%d'),
            'second_day_date': dates[1].strftime('%Y-%m-%d') if len(dates) > 1 else "N/A",
            'Symbol': ticker_value,
            'Name': f"{ticker_value} Stock",
            'Last_Sale': f"${df['Close'].iloc[-1]:.2f}",
            'Net_Change': f"${(df['Close'].iloc[-1] - df['Open'].iloc[-1]):.2f}",
            'Percent_Change': f"{((df['Close'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1] * 100):.2f}%",
            'Volume': f"{int(df['Volume'].iloc[-1]):,}",
            'Market_Cap': "N/A",
            'Country': "USA",
            'IPO_Year': "N/A",
            'Sector': "Technology",
            'Industry': "Technology",
            'predictions': [f"${p:.2f}" for p in predictions],
            'prediction_dates': [d.strftime('%Y-%m-%d') for d in dates],
            'avg_prediction': f"${np.mean(predictions):.2f}",
            'min_prediction': f"${min(predictions):.2f}",
            'max_prediction': f"${max(predictions):.2f}",
        }

        print("Prediction completed successfully")
        return render(request, "result.html", context=context)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return render(request, 'API_Down.html', {})

# ...rest of the existing code...
