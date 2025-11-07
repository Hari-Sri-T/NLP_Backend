# services/sdg_llm.py
import os
import json
import logging
import requests

GEMINI_MODEL = "gemini-flash-lite-latest"

def call_gemini_api(prompt: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in environment variables.")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()

        # ✅ Correct path for Gemini Flash Lite
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    except Exception as e:
        logging.error(f"[Gemini Error] {e} - Response: {getattr(response, 'text', '')}")
        return None


SYSTEM_PROMPT = (
    "You are an SDG/ESG financial impact analyst.\n"
    "Your task is to assess the overall SDG (Sustainable Development Goals) impact of the given news article.\n\n"
    "Consider all 17 SDGs. Based on the article, determine if the company's actions are Positive, Negative, or Neutral towards these goals.\n"
    "- **Positive:** The news describes a clear, positive contribution to one or more SDGs.\n"
    "- **Negative:** The news describes actions that clearly undermine one or more SDGs.\n"
    "- **Neutral:** The news has no clear relevance to any of the 17 SDGs.\n\n"
    "Based on this, provide a single overall score from 0 (very negative) to 100 (very positive), where 50 is perfectly neutral.\n\n"
    "Respond ONLY in the following JSON format. Do not add any other text, markdown, or explanations outside the JSON structure.\n"
    '{ "label": "Positive|Neutral|Negative", "score": 0-100, "explanation": "A short summary of the SDG impact." }\n'
)

def classify_news(text: str):
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"News:\n{text}\n\n"
        "Respond ONLY in JSON (no comments, no markdown):\n"
        '{"label": "", "score": 0, "explanation": ""}'
    )

    raw_output = call_gemini_api(prompt)

    if not raw_output or raw_output.strip() == "":
        return {
            "label": "Neutral",
            "score": 50,
            "explanation": "No SDG-related impact detected."
        }

    try:
        data = json.loads(raw_output)

        # ✅ Validate fields
        score = int(data.get("score", 50))
        score = max(0, min(100, score))

        label = data.get("label", "Neutral")
        if label not in ("Positive", "Neutral", "Negative"):
            label = "Neutral"

        return {
            "label": label,
            "score": score,
            "explanation": data.get("explanation", "")
        }

    except json.JSONDecodeError:
        logging.error(f"[Gemini JSON Error] Failed to parse output: {raw_output}")
        return {
            "label": "Neutral",
            "score": 50,
            "explanation": "Invalid JSON received from model."
        }