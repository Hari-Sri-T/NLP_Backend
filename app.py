import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import yfinance as yf
import requests

# --- NEW IMPORTS (from your new file) ---
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager
from datetime import timedelta

# --- ADDED SDG LOGIC (1 of 3): Import classify_news ---
from services.sdg_llm import classify_news
# ----------------------------------------------------

from services.statistical_model import get_advanced_metrics
from services.scoring import get_price_trend_score, get_valuation_score, combine_scores, map_to_recommendation

app = Flask(__name__)
CORS(app)

# --- NEW APP CONFIGURATION ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db' 
app.config['JWT_SECRET_KEY'] = 'super-secret-key-please-change-me' 
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=30) 

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
# -----------------------------

# --- ADD THIS CODE ---
@app.before_request
def create_tables():
    # This runs ONCE before the first request.
    # It checks if the tables are already there and creates them if not.
    db.create_all()
# ---

# --- NEW USER DATABASE MODEL ---
class User(db.Model): 
    id = db.Column(db.Integer, primary_key=True) 
    username = db.Column(db.String(80), unique=True, nullable=False) 
    password = db.Column(db.String(120), nullable=False) 

    def __repr__(self): 
        return f'<User {self.username}>' 
# -------------------------------

# --- NEW BOOKMARK DATABASE MODEL ---
class Bookmark(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stock_symbol = db.Column(db.String(20), nullable=False)
    # Make sure a user can't bookmark the same stock twice
    __table_args__ = (db.UniqueConstraint('user_id', 'stock_symbol', name='_user_stock_uc'),)
# -------------------------------


# --- NEW: First-time database setup command ---
@app.cli.command("create_db") 
def create_db(): 
    with app.app_context(): 
        db.create_all() 
        print("Database tables created!") 
# ----------------------------------------------


@app.route("/") 
def home(): 
    # This is correct for a React Native backend
    return jsonify({"message": "Backend is running"}) 

logging.basicConfig( 
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
)

# --- NEW: /register endpoint ---
@app.route("/register", methods=["POST"]) 
def register(): 
    data = request.get_json() 
    username = data.get('username') 
    password = data.get('password') 

    if not username or not password: 
        return jsonify({"error": "Username and password are required"}), 400 

    user_exists = User.query.filter_by(username=username).first() 
    if user_exists: 
        return jsonify({"error": "Username already exists"}), 409 

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8') 
    new_user = User(username=username, password=hashed_password) 
    
    try: 
        db.session.add(new_user) 
        db.session.commit() 
        return jsonify({"message": f"User {username} created successfully"}), 201 
    except Exception as e: 
        db.session.rollback() 
        logging.error(f"Error creating user: {e}", exc_info=True) 
        return jsonify({"error": "Internal server error"}), 500 

# --- NEW: /login endpoint ---
@app.route("/login", methods=["POST"]) 
def login(): 
    data = request.get_json() 
    username = data.get('username') 
    password = data.get('password') 

    if not username or not password: 
        return jsonify({"error": "Username and password are required"}), 400 

    user = User.query.filter_by(username=username).first() 

    if user and bcrypt.check_password_hash(user.password, password): 
        # Passwords match! Create a token.
        access_token = create_access_token(identity=user.username) 
        return jsonify(access_token=access_token), 200 
    else: 
        # Invalid credentials
        return jsonify({"error": "Invalid username or password"}), 401 

# --- /stocks endpoint (search/autocomplete)
@app.route("/stocks", methods=["GET"]) 
def get_stocks(): 
    from services.finnhub_service import search_stocks, get_company_logo, finnhub_client 

    if not finnhub_client: 
        error_message = "Finnhub client not initialized. Check FINNHUB_API_KEY in the backend env." 
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
                if not symbol or symbol in processed_symbols: 
                    continue 
                if item.get("type") != "Common Stock": 
                    continue 
                if '.' in symbol: 
                    continue 
                stocks_to_fetch.append(symbol) 
                processed_symbols.add(symbol) 

        for symbol in stocks_to_fetch: 
            if not symbol: 
                continue 
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
        error_message = f"Server error while fetching stocks: {e}" 
        logging.error(error_message, exc_info=True) 
        return jsonify({"error": error_message}), 500 


# --- /history endpoint (for chart)
@app.route("/history", methods=["GET"]) 
def get_history(): 
    symbol = request.args.get("symbol") 
    period = request.args.get("period", "1y")  # default 1 year 

    if not symbol: 
        return jsonify({"error": "Stock symbol is required."}), 400 

    try: 
        stock = yf.Ticker(symbol) 
        hist = stock.history(period=period, interval="1d") 
        if hist.empty: 
            return jsonify({"error": f"No historical data for {symbol} with period {period}."}), 404 

        hist.reset_index(inplace=True) 
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d') 

        data = { 
            'dates': hist['Date'].tolist(), 
            'prices': hist['Close'].tolist() 
        } 
        return jsonify(data) 
    except Exception as e: 
        logging.error(f"Error fetching history for {symbol}: {e}", exc_info=True) 
        return jsonify({"error": str(e)}), 500 

# --- (Helper functions: call_gemini_api, etc.) ---
def call_gemini_api(prompt): 
    gemini_api_key = os.environ.get("GEMINI_API_KEY") 
    if not gemini_api_key: 
        logging.warning("GEMINI_API_KEY not set.") 
        return None 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={gemini_api_key}" 
    headers = {'Content-Type': 'application/json'} 
    data = {"contents": [{"parts": [{"text": prompt}]}]} 
    try: 
        response = requests.post(url, headers=headers, json=data, timeout=15) 
        response.raise_for_status() 
        result = response.json() 
        return result['candidates'][0]['content']['parts'][0]['text'].strip() 
    except Exception as e: 
        logging.error(f"Gemini API request/parse failed: {e}") 
        return None 

def analyze_sentiment_with_gemini(news_text: str) -> int: 
    if not news_text or not news_text.strip(): 
        return 50 
    prompt = f"""
    You are a financial analyst. Based on this news, rate the sentiment 0–100 where 0=Strong Sell and 100=Strong Buy. Only output the number.
    News: {news_text}
    """ 
    response_text = call_gemini_api(prompt) 
    try: 
        score = int(response_text) 
        return max(0, min(100, score)) 
    except Exception: 
        logging.error(f"Could not parse sentiment score: {response_text}") 
        return 50 

def summarize_news_with_llm(symbol, articles): 
    if not articles: 
        return "No significant news found." 
    news_text = "\n\n---\n\n".join([f"Headline: {a.get('title','')}\nSummary: {a.get('description','')}" for a in articles if a.get('title')]) 
    if not news_text.strip(): 
        return "No news content available to summarize." 
    prompt = f"""
    You are a financial analyst. Create concise bullet-point summary of most important financial news for '{symbol}' from the articles below.
    Articles:
    {news_text}
    """ 
    summary = call_gemini_api(prompt) 
    return summary or "News summary could not be generated due to an API error." 

def generate_explanation(symbol, last_close, predicted_close, trend_score, 
                         sentiment_score, final_score, recommendation, news_text): 
    prompt = f"""
    Explain in 3–5 sentences why the recommendation for {symbol} is "{recommendation}".
    Use only the provided data.
    """ 
    explanation = call_gemini_api(prompt) 
    return explanation or "Explanation not available due to an API error." 
# --- (End of helper functions) ---


# --- /analyze endpoint (main analysis) ---
@app.route("/analyze", methods=["GET"]) 
@jwt_required()  # <-- This is the lock! 
def analyze(): 
    current_user_username = get_jwt_identity() 
    logging.info(f"Authorized user '{current_user_username}' requesting analysis.") 
    
    from services.stock_service import get_stock_data, predict_next_close 
    from services.news_service import get_company_news 
    from services.finnhub_service import get_company_logo 

    symbol = request.args.get("symbol", "GOOG") 
    try: 
        logging.info(f"Starting analysis for {symbol}") 

        df = get_stock_data(symbol) 
        if df.empty: 
            raise ValueError(f"Could not retrieve stock data for {symbol}.") 

        last_close = float(df["close"].iloc[-1]) 
        predicted_close = predict_next_close(symbol) 
        trend_score = get_price_trend_score(predicted_close, last_close) 

        company_profile = get_company_logo(symbol, get_full_profile=True) 
        news_query = company_profile.get('name') if company_profile and company_profile.get('name') else symbol 

        articles = get_company_news(news_query) 
        news_text = " ".join([ (a.get('title','') + " " + a.get('description','')) for a in articles if a.get('description')]) 
        sentiment_score = analyze_sentiment_with_gemini(news_text) 
        news_summary = summarize_news_with_llm(symbol, articles) 

        # --- ADDED SDG LOGIC (2 of 3): Classify news and get score ---
        sdg_result = classify_news(news_summary)
        sdg_score = sdg_result.get("score", 50)
        sdg_label = sdg_result.get("label", "Neutral")
        sdg_explanation = sdg_result.get("explanation", "No SDG-related impact detected.")
        # -----------------------------------------------------------

        advanced_metrics = get_advanced_metrics(symbol) 
        if not advanced_metrics: 
            logging.warning("Advanced metrics missing; continuing with defaults.") 
            advanced_metrics = {} 

        graham_number = advanced_metrics.get("valuation_models", {}).get("graham_number", 0) 
        valuation_score = get_valuation_score(last_close, graham_number) 

        # Pass the new sdg_score to the combine function
        final_score = combine_scores(
            trend_score, sentiment_score, valuation_score, sdg_score=sdg_score
        ) 
        recommendation = map_to_recommendation(final_score, trend_score=trend_score, sentiment_score=sentiment_score) 

        explanation = generate_explanation(symbol, last_close, predicted_close, trend_score, sentiment_score, final_score, recommendation, news_text) 

        result = { 
            "symbol": symbol, 
            "last_close": last_close, 
            "predicted_close": predicted_close, 
            "recommendation": recommendation, 
            "final_score": final_score, 
            "scores": {"trend": trend_score, "sentiment": sentiment_score, "valuation": valuation_score}, 
            
            # --- ADDED SDG LOGIC (3 of 3): Add sdg block to JSON response ---
            "sdg": {
                "label": sdg_label,
                "score": sdg_score,
                "explanation": sdg_explanation
            },
            # ----------------------------------------------------------------
            
            "explanation": explanation, 
            "latest_news_summary": news_summary, 
            "advanced_metrics": advanced_metrics 
        } 
        logging.info(f"Finished analysis for {symbol}") 
        return jsonify(result) 
    except Exception as e: 
        error_message = f"Error during analysis for {symbol}: {e}" 
        logging.error(error_message, exc_info=True) 
        return jsonify({"error": error_message}), 500 


# --- simple health endpoint
@app.route("/health", methods=["GET"]) 
def health(): 
    return jsonify({"status": "ok"}), 200 

# --- /search endpoint (for frontend search bar)
@app.route("/search", methods=["GET"]) 
def search(): 
    from services.finnhub_service import search_stocks, get_company_logo, finnhub_client 

    query = request.args.get("query", "").strip() 
    if not query: 
        return jsonify([]) 

    if not finnhub_client: 
        logging.error("Finnhub client not initialized. Check FINNHUB_API_KEY.") 
        return jsonify({"error": "Backend misconfigured. FINNHUB_API_KEY missing."}), 500 

    try: 
        raw_results = search_stocks(query) 
        simplified = [] 

        for r in raw_results: 
            if not r.get("symbol") or "." in r["symbol"]: 
                continue 
            if r.get("type") != "Common Stock": 
                continue 
            logo = None 
            try: 
                profile = get_company_logo(r["symbol"], get_full_profile=True) 
                logo = profile.get("logo") if profile else None 
            except Exception: 
                logo = None 
            simplified.append({ 
                "symbol": r["symbol"], 
                "description": r.get("description", ""), 
                "logo": logo 
            }) 
        logging.info(f"Search results for '{query}': {len(simplified)} stocks found.") 
        return jsonify(simplified) 
    except Exception as e: 
        logging.error(f"Error during search for query '{query}': {e}", exc_info=True) 
        return jsonify({"error": str(e)}), 500 

# --- /chart endpoint (for line chart data)
@app.route("/chart", methods=["GET"]) 
def get_chart(): 
    import yfinance as yf 
    import datetime as dt 

    symbol = request.args.get("symbol") 
    if not symbol: 
        return jsonify({"error": "Stock symbol required"}), 400 
    try: 
        end = dt.datetime.now() 
        start = end - dt.timedelta(days=365) 
        stock = yf.Ticker(symbol) 
        hist = stock.history(start=start, end=end) 
        data = [ 
            {"time": int(row[0].timestamp()), "price": float(row[1]["Close"])} 
            for row in hist.iterrows() 
        ] 
        return jsonify(data) 
    except Exception as e: 
        print("Error fetching chart:", e) 
        return jsonify({"error": str(e)}), 500 

# --- NEW BOOKMARK ENDPOINTS ---

@app.route("/bookmarks", methods=["GET"])
@jwt_required()
def get_bookmarks():
    current_user_username = get_jwt_identity()
    user = User.query.filter_by(username=current_user_username).first()
    
    if not user:
        return jsonify({"error": "User not found"}), 404

    bookmarks = Bookmark.query.filter_by(user_id=user.id).all()
    # Just return a simple list of symbols
    symbols = [b.stock_symbol for b in bookmarks]
    return jsonify(symbols)


@app.route("/bookmark", methods=["POST"])
@jwt_required()
def add_bookmark():
    current_user_username = get_jwt_identity()
    user = User.query.filter_by(username=current_user_username).first()
    
    data = request.get_json()
    symbol = data.get('symbol')

    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400

    # Check if it's already bookmarked
    exists = Bookmark.query.filter_by(user_id=user.id, stock_symbol=symbol).first()
    if exists:
        return jsonify({"message": "Already bookmarked"}), 200

    new_bookmark = Bookmark(user_id=user.id, stock_symbol=symbol)
    try:
        db.session.add(new_bookmark)
        db.session.commit()
        return jsonify({"message": "Bookmark added"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/bookmark/<string:symbol>", methods=["DELETE"])
@jwt_required()
def delete_bookmark(symbol):
    current_user_username = get_jwt_identity()
    user = User.query.filter_by(username=current_user_username).first()

    bookmark = Bookmark.query.filter_by(user_id=user.id, stock_symbol=symbol).first()
    
    if not bookmark:
        return jsonify({"error": "Bookmark not found"}), 404

    try:
        db.session.delete(bookmark)
        db.session.commit()
        return jsonify({"message": "Bookmark removed"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# --- END OF NEW ENDPOINTS ---

if __name__ == "__main__": 
    # This is the correct way to run for React Native development
    app.run(host="0.0.0.0", port=5000, debug=True)