def get_price_trend_score(predicted_close, last_close):
    pct_change = ((predicted_close - last_close) / last_close) * 100
    if pct_change > 2: return 90
    elif pct_change > 0.5: return 70
    elif pct_change > -0.5: return 50
    elif pct_change > -2: return 30
    else: return 10

def get_valuation_score(current_price, graham_number):
    if graham_number is None or current_price is None:
        return 50
    if current_price < graham_number * 0.75:
        return 90
    elif current_price < graham_number:
        return 75
    elif current_price < graham_number * 1.25:
        return 50
    elif current_price < graham_number * 1.5:
        return 30
    else:
        return 10

def combine_scores(trend_score, sentiment_score, valuation_score, sdg_score=50,
                   w_trend=0.25, w_sentiment=0.25, w_valuation=0.20, w_sdg=0.30):
    """
    Combines trend, sentiment, valuation, AND SDG score with new weights.
    """
    return (
        w_trend * trend_score +
        w_sentiment * sentiment_score +
        w_valuation * valuation_score +
        w_sdg * sdg_score
    )

def map_to_recommendation(final_score, trend_score=None, sentiment_score=None):
    if (sentiment_score is not None and sentiment_score >= 85 and final_score >= 60) or \
       (trend_score is not None and trend_score >= 85 and final_score >= 60):
        return "Strong Buy"

    if final_score >= 70: return "Buy"
    elif final_score >= 40: return "Hold"
    else: return "Sell"
    