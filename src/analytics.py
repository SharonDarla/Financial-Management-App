# voice_analytics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import speech_recognition as sr
import pyttsx3

# ------------------------------
# Load Data
# ------------------------------
data = pd.read_csv("data20.csv")
data['Date'] = pd.to_datetime(data['Date'])

# Ensure plots folder exists
os.makedirs("plots", exist_ok=True)

# ------------------------------
# Analytics Functions
# ------------------------------
def max_price(company):
    return data[data['Company']==company]['Close'].max()

def min_price(company):
    return data[data['Company']==company]['Close'].min()

def average_price(company):
    return data[data['Company']==company]['Close'].mean()

def volatility(company):
    return data[data['Company']==company]['Close'].pct_change().std()

def daily_returns(company):
    return data[data['Company']==company]['Close'].pct_change().mean()

def cagr(company):
    df = data[data['Company']==company].sort_values('Date')
    start = df['Close'].iloc[0]
    end = df['Close'].iloc[-1]
    n = len(df)/252
    return ((end/start)**(1/n) - 1)*100

def moving_average(company, window=20):
    return data[data['Company']==company]['Close'].rolling(window).mean().iloc[-1]

def ema(company, span=20):
    return data[data['Company']==company]['Close'].ewm(span=span, adjust=False).mean().iloc[-1]

def rsi(company, periods=14):
    df = data[data['Company']==company]['Close']
    delta = df.diff()
    gain = delta.where(delta>0, 0)
    loss = -delta.where(delta<0, 0)
    avg_gain = gain.rolling(periods).mean().iloc[-1]
    avg_loss = loss.rolling(periods).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss !=0 else 0
    return 100 - (100/(1+rs))

def macd(company, fast=12, slow=26):
    df = data[data['Company']==company]['Close']
    ema_fast = df.ewm(span=fast, adjust=False).mean()
    ema_slow = df.ewm(span=slow, adjust=False).mean()
    return ema_fast.iloc[-1] - ema_slow.iloc[-1]

analytics_dict = {
    "max price": max_price,
    "min price": min_price,
    "average price": average_price,
    "volatility": volatility,
    "daily return": daily_returns,
    "cagr": cagr,
    "moving average": moving_average,
    "ema": ema,
    "rsi": rsi,
    "macd": macd
}

def get_analytics(company, query):
    func = analytics_dict.get(query.lower())
    if func:
        return func(company)
    else:
        return "Analytics not found"

# ------------------------------
# Voice Functions
# ------------------------------
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    print(f"🗣️ {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("\n🎤 Listening... Ask your query (e.g., 'RSI of AAPL')")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"🔍 You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        speak("Sorry, I could not understand. Please try again.")
        return None
    except sr.RequestError:
        speak("Speech recognition service is unavailable.")
        return None

def extract_company_and_metric(query):
    companies = [c.lower() for c in data['Company'].unique()]
    metrics = [m.lower() for m in analytics_dict.keys()]
    company = None
    metric = None

    for comp in companies:
        if comp in query:
            company = comp
            break

    for met in metrics:
        if met in query:
            metric = met
            break

    # Synonyms
    if not metric:
        if "rsi" in query:
            metric = "rsi"
        elif "macd" in query:
            metric = "macd"
        elif "average" in query:
            metric = "average price"
        elif "max" in query or "highest" in query:
            metric = "max price"
        elif "min" in query or "lowest" in query:
            metric = "min price"
        elif "volatility" in query:
            metric = "volatility"
        elif "return" in query:
            metric = "daily return"
        elif "ema" in query:
            metric = "ema"
        elif "moving average" in query:
            metric = "moving average"
        elif "cagr" in query:
            metric = "cagr"
        elif "histogram" in query:
            metric = "histogram"
    return company, metric

# ------------------------------
# Advanced Analytics / Graphs
# ------------------------------
def plot_metric(company, metric):
    df = data[data['Company']==company].sort_values('Date')
    plt.figure(figsize=(10,5))
    sns.set_style("whitegrid")

    if metric in ["moving average", "ema"]:
        df["MA20"] = df['Close'].rolling(20).mean()
        df["EMA20"] = df['Close'].ewm(span=20, adjust=False).mean()
        plt.plot(df['Date'], df['Close'], label='Close')
        plt.plot(df['Date'], df["MA20"], label='MA20')
        plt.plot(df['Date'], df["EMA20"], label='EMA20')
        plt.title(f"{company} - Price with MA & EMA")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend()

    elif metric == "rsi":
        delta = df['Close'].diff()
        gain = delta.where(delta>0, 0)
        loss = -delta.where(delta<0,0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100/(1+rs))
        plt.plot(df['Date'], df['RSI'], label='RSI', color='orange')
        plt.axhline(70, color='red', linestyle='--')
        plt.axhline(30, color='green', linestyle='--')
        plt.title(f"{company} - RSI")
        plt.ylabel("RSI")
        plt.xlabel("Date")
        plt.legend()

    elif metric == "macd":
        ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal = macd_line.ewm(span=9, adjust=False).mean()
        plt.plot(df['Date'], macd_line, label='MACD')
        plt.plot(df['Date'], signal, label='Signal', linestyle='--')
        plt.title(f"{company} - MACD")
        plt.ylabel("Value")
        plt.xlabel("Date")
        plt.legend()

    elif metric == "daily return":
        df['DailyReturn'] = df['Close'].pct_change()
        plt.plot(df['Date'], df['DailyReturn'], label='Daily Returns', color='purple')
        plt.title(f"{company} - Daily Returns")
        plt.ylabel("Return")
        plt.xlabel("Date")
        plt.legend()

    elif metric == "volatility":
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()
        plt.plot(df['Date'], df['Volatility'], label='Volatility', color='brown')
        plt.title(f"{company} - Rolling Volatility")
        plt.ylabel("Volatility")
        plt.xlabel("Date")
        plt.legend()

    elif metric == "cagr":
        plt.plot(df['Date'], df['Close'].pct_change().cumsum(), label='Cumulative Returns')
        plt.title(f"{company} - CAGR/Returns Trend")
        plt.ylabel("Cumulative Returns")
        plt.xlabel("Date")
        plt.legend()

    elif metric in ["max price", "min price", "average price"]:
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        plt.axhline(get_analytics(company, metric), color='red', linestyle='--', label=f'{metric}')
        plt.title(f"{company} - {metric.title()}")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend()

    elif metric == "histogram":
        df['Returns'] = df['Close'].pct_change()
        sns.histplot(df['Returns'].dropna(), bins=30, kde=True, color='teal')
        plt.title(f"{company} - Return Distribution")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")

    else:
        plt.plot(df['Date'], df['Close'], label='Close Price')
        plt.title(f"{company} - Close Price")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend()

    filename = f"plots/{company}_{metric.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

# ------------------------------
# Main Voice-Analytics Loop
# ------------------------------
def main():
    speak("Hello! I can provide analytics insights with charts. Say 'exit' to quit.")
    print("Say 'exit' or 'quit' to stop.\n")

    while True:
        query = listen()
        if not query:
            continue
        if "exit" in query or "quit" in query or "stop" in query:
            speak("Goodbye!")
            break

        company, metric = extract_company_and_metric(query)
        if not company:
            speak("I couldn't detect the company name. Please try again.")
            continue
        if not metric:
            speak("I couldn't detect which metric you want. Please try again.")
            continue

        company_name = next((c for c in data['Company'].unique() if c.lower() == company), company)
        result = get_analytics(company_name, metric)

        if isinstance(result, (int, float)):
            speak(f"The {metric} for {company_name} is {round(result,2)}")
        else:
            speak(f"Sorry, could not fetch {metric} for {company_name}")
        print(f"✅ {metric.title()} for {company_name}: {result}")

        # Generate graph for visual analytics
        graph_file = plot_metric(company_name, metric)
        speak(f"A graph for {metric} of {company_name} has been saved as {graph_file}")

if __name__ == "__main__":
    main()
