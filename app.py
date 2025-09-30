import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import yfinance as yf
import pandas as pd

from services.statistical_model import get_advanced_metrics
from services.scoring import get_price_trend_score, get_valuation_score, combine_scores, map_to_recommendation

import requests
import json


app = Flask(__name__)
CORS(app)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@app.route("/stocks", methods=["GET"])
def get_stocks():
    """
    Returns filtered stock search results with robust error handling.
    """
    from services.finnhub_service import search_stocks, get_company_logo, finnhub_client
    
    if not finnhub_client:
        error_message = "Finnhub client not initialized. Please check if the FINNHUB_API_KEY is set correctly in the backend environment."
        logging.error(error_message)
        return jsonify({"error": error_message}), 500

    query = request.args.get("q", "").strip()
    results = []
    processed_symbols = set()

    try:
        stocks_to_fetch = []
        if ',' in query and query:
            stocks_to_fetch = [s.strip() for s in query.split(',')]
        elif query:
            search_results = search_stocks(query)
            for item in search_results:
                symbol = item.get("symbol")
                if not symbol or symbol in processed_symbols: continue
                if item.get("type") != "Common Stock": continue
                if '.' in symbol: continue
                stocks_to_fetch.append(symbol)
                processed_symbols.add(symbol)
        
        for symbol in stocks_to_fetch:
            if not symbol: continue
            try:
                profile = get_company_logo(symbol, get_full_profile=True)
                if profile and profile.get('name'):
                     results.append({
                        "symbol": symbol,
                        "name": profile.get('name'),
                        "logo": profile.get('logo')
                    })
            except Exception as e:
                logging.warning(f"Could not fetch full profile for {symbol}, skipping: {e}")

        return jsonify(results)

    except Exception as e:
        error_message = f"An unexpected error occurred on the server while fetching stocks: {e}"
        logging.error(error_message, exc_info=True)
        return jsonify({"error": error_message}), 500

# --- NEW ENDPOINT FOR HISTORICAL DATA ---
@app.route("/history", methods=["GET"])
def get_history():
    """
    Fetches historical stock data for a given symbol and period.
    """
    symbol = request.args.get("symbol")
    period = request.args.get("period", "1y") # Default to 1 year

    if not symbol:
        return jsonify({"error": "Stock symbol is required."}), 400

    try:
        # Fetch data using yfinance
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval="1d")

        if hist.empty:
            return jsonify({"error": f"No historical data found for symbol {symbol} with period {period}."}), 404

        # Reset index to make 'Date' a column
        hist.reset_index(inplace=True)
        
        # Convert timestamps to a more readable format (YYYY-MM-DD)
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')

        # Prepare data for JSON response
        data = {
            'dates': hist['Date'].tolist(),
            'prices': hist['Close'].tolist()
        }
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error fetching history for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Helper function to make direct calls to the Gemini API
def call_gemini_api(prompt):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={gemini_api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
        
        result = response.json()
        
        # THE FIX IS HERE: The path to the text is different in the actual response
        return result['candidates'][0]['content']['parts'][0]['text'].strip()

    except requests.exceptions.RequestException as e:
        logging.error(f"Gemini API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"Failed to parse Gemini API response: {e} | Response: {response.text}")
        return None

# Replace your old analyze_sentiment_with_gemini function
def analyze_sentiment_with_gemini(news_text: str) -> int:
    if not news_text or not news_text.strip():
        return 50

    prompt = f"""
    You are a financial analyst. Based on the following news, rate the sentiment for the stock on a scale of 0–100, where 0 = Strong Sell, 50 = Hold, 100 = Strong Buy. Only output the number.
    News: {news_text}
    """
    response_text = call_gemini_api(prompt)
    try:
        score = int(response_text)
        return max(0, min(100, score))
    except (ValueError, TypeError):
        logging.error(f"Could not parse sentiment score from Gemini response: {response_text}")
        return 50

# Replace your old summarize_news_with_llm function
def summarize_news_with_llm(symbol, articles):
    if not articles:
        return "No significant news found."
    
    news_text = "\n\n---\n\n".join([f"Headline: {a['title']}\nSummary: {a.get('description', '')}" for a in articles if a.get('title')])
    if not news_text.strip():
        return "No news content available to summarize."
        
    prompt = f"""
    You are a financial analyst. Your task is to create a concise, bullet-point summary of the most important financial news for the company '{symbol}' from the articles provided below.
    - Identify articles *directly* about '{symbol}'s business or financial performance.
    - Ignore irrelevant news (politics, other companies, etc.).
    - Summarize key takeaways in this format: - **News Event:** [Event]. **Financial Implication:** [Effect].
    - If no relevant news is found, respond with ONLY this sentence: "No significant company-specific news was found in the latest articles."

    Articles to analyze:
    {news_text}
    """
    summary = call_gemini_api(prompt)
    return summary or "News summary could not be generated due to an API error."

# Replace your old generate_explanation function
def generate_explanation(symbol, last_close, predicted_close, trend_score,
                         sentiment_score, final_score, recommendation, news_text):
    prompt = f"""
    You are a stock analyst assistant. Explain in 3–5 sentences max why the recommendation for {symbol} is "{recommendation}".
    Use only the provided data:
    - Last Close Price: ${last_close:.2f}
    - AI Predicted Next Close: ${predicted_close:.2f}
    - Price Trend Score: {trend_score}
    - News Sentiment Score: {sentiment_score}
    - Final AI Score: {final_score}
    - Key News Summary: {news_text[:400]}

    Rules:
    - Do NOT invent details. Be factual and concise.
    - Use clear bullet points.
    """
    explanation = call_gemini_api(prompt)
    return explanation or "Explanation not available due to an API error."

@app.route("/analyze", methods=["GET"])
def analyze():
    from services.stock_service import get_stock_data, predict_next_close
    from services.news_service import get_company_news
   
    symbol = request.args.get("symbol", "GOOG")
    try:
        logging.info(f"--- Starting full analysis for {symbol} ---")

        # 1. Get Price Data & Prediction
        df = get_stock_data(symbol)
        
           # ADD THIS BLOCK TO CHECK FOR EMPTY DATA
        if df.empty:
            logging.error(f"Failed to download stock data for {symbol}. DataFrame is empty, likely due to a timeout or invalid symbol.")
            raise ValueError(f"Could not retrieve stock data for {symbol}. Please check the symbol or try again later.")


        last_close = float(df["close"].iloc[-1])
        predicted_close = predict_next_close(df)
        trend_score = get_price_trend_score(predicted_close, last_close)
        logging.info(f"Last close={last_close}, Predicted={predicted_close}, TrendScore={trend_score}")


        # 2. Get News & Sentiment
        from services.news_service import get_company_news
        # Import the function to get the company profile
        from services.finnhub_service import get_company_logo 
        
        # Get company profile to find the full name for a better news search
        company_profile = get_company_logo(symbol, get_full_profile=True)
        news_query = symbol # Default to the symbol
        if company_profile and company_profile.get('name'):
            news_query = company_profile['name'] # Use full name if available
            logging.info(f"Using full company name for news search: {news_query}")
        else:
            logging.info(f"Could not find company name for {symbol}, falling back to symbol for news search.")
        
        articles = get_company_news(news_query) # Use the best available query
        
        news_text = " ".join([a['title'] + " " + a['description'] for a in articles if a.get('description')])
        sentiment_score = analyze_sentiment_with_gemini(news_text)
        news_summary = summarize_news_with_llm(symbol, articles)
        logging.info(f"Sentiment score: {sentiment_score}")

        # 3. Get Advanced Metrics (Valuation, Risk, Daily Data)
        advanced_metrics = get_advanced_metrics(symbol)
        if not advanced_metrics:
            raise ValueError("Could not retrieve advanced metrics.")
        
        graham_number = advanced_metrics["valuation_models"]["graham_number"]
        valuation_score = get_valuation_score(last_close, graham_number)
        logging.info(f"Graham Number={graham_number}, Valuation Score={valuation_score}")

        # 4. Combine Scores & Get Recommendation
        final_score = combine_scores(trend_score, sentiment_score, valuation_score)
        recommendation = map_to_recommendation(final_score, trend_score=trend_score, sentiment_score=sentiment_score)
        logging.info(f"Final score: {final_score}, Recommendation: {recommendation}")

        # 5. Generate Explanation
        explanation = generate_explanation(symbol, last_close, predicted_close, trend_score, sentiment_score, final_score, recommendation, news_text)
        
        # 6. Assemble Final Result
        result = {
            "symbol": symbol,
            "last_close": last_close,
            "predicted_close": predicted_close,
            "recommendation": recommendation,
            "final_score": final_score,
            "scores": {"trend": trend_score, "sentiment": sentiment_score, "valuation": valuation_score},
            "explanation": explanation,
            "latest_news_summary": news_summary,
            "advanced_metrics": advanced_metrics # Nest all the new data here
        }
        logging.info(f"--- Finished analysis for {symbol} ---")
        return jsonify(result)

    except Exception as e:
        error_message = f"An error occurred during analysis for {symbol}: {e}"
        logging.error(error_message, exc_info=True)
        return jsonify({"error": error_message}), 500
    
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

