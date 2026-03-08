# app.py (FULL - UPDATED: LabelEncoder HATASI FIX + GEMINI ENTEGRASYONU)

import re
import json
import traceback
import os
from typing import Dict, Any, Tuple

import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from google import genai

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__, template_folder="templates")

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "model.h5"
ENCODERS_PATH = "encoders.pkl"
SCALER_PATH = "scaler.pkl"
THRESHOLDS_PATH = "thresholds.pkl"

# -----------------------------
# Load artifacts
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)        
scaler = joblib.load(SCALER_PATH)            
thresholds = joblib.load(THRESHOLDS_PATH)    

# FIX: "Books" kategorisi eklendi (Eğitim sırasıyla birebir eşleşmeli)
CATEGORIES = ["Books", "Clothing", "Cosmetics", "Food & Beverage", "Shoes", "Souvenir", "Technology", "Toys"]

# -----------------------------
# Gemini Kurulumu
# -----------------------------
# GITHUB KULLANICILARI İÇİN NOT: Aşağıdaki tırnak içine kendi Gemini API Key'inizi yazın
GEMINI_API_KEY = "Buraya_Kendi_Api_keyini_yaz" 
client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_EXTRACTION_PROMPT = '''
Kullanıcıdan gelen alışveriş senaryosunu analiz et.

SADECE aşağıdaki JSON formatında cevap ver.
Açıklama, yorum, metin veya markdown ekleme.

JSON alanları:
- age: integer
- gender: "Male" veya "Female"
- shopping_mall: alışveriş merkezi adı
- payment_method: "Cash", "Credit Card" veya "Debit Card"
- day_type: "Weekday" veya "Weekend"
- price: integer
'''

# -----------------------------
# Helpers
# -----------------------------
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def normalize_gender(g: str) -> str:
    if not g:
        return ""
    g2 = str(g).strip().lower()
    if g2 in ["male", "m", "erkek", "man"]:
        return "Male"
    if g2 in ["female", "f", "kadın", "kadin", "woman"]:
        return "Female"
    return str(g).strip()


def encoder_get_id(encoder_obj, key: str, fallback_to_zero=True) -> int:
    if encoder_obj is None:
        return 0 if fallback_to_zero else -1

    if isinstance(encoder_obj, dict):
        if key in encoder_obj:
            return int(encoder_obj[key])

        key_l = str(key).strip().lower()
        for k, v in encoder_obj.items():
            if str(k).strip().lower() == key_l:
                return int(v)
        return 0 if fallback_to_zero else -1

    if hasattr(encoder_obj, "transform") and hasattr(encoder_obj, "classes_"):
        try:
            return int(encoder_obj.transform([key])[0])
        except Exception:
            key_l = str(key).strip().lower()
            classes = list(getattr(encoder_obj, "classes_", []))
            for c in classes:
                if str(c).strip().lower() == key_l:
                    return int(encoder_obj.transform([c])[0])
            return 0 if fallback_to_zero else -1

    return 0 if fallback_to_zero else -1


def today_yyyy_mm_dd() -> str:
    import datetime as _dt
    return _dt.date.today().isoformat()


def parse_invoice_date_or_default(form: Dict[str, Any], extracted: Dict[str, Any] | None = None) -> str:
    ds = None
    if extracted:
        ds = extracted.get("invoice_date") or extracted.get("date") or extracted.get("invoiceDate")

    if not ds:
        ds = (
            form.get("invoice_date")
            or form.get("date")
            or form.get("invoiceDate")
            or form.get("invoice_date_input")
            or ""
        )

    if not ds:
        return today_yyyy_mm_dd()

    ds = str(ds).strip()
    if len(ds) >= 10:
        ds = ds[:10]

    if len(ds) != 10 or ds[4] != "-" or ds[7] != "-":
        return today_yyyy_mm_dd()

    return ds


# YENİ: Regex yerine Gemini ile metin çıkarma fonksiyonu
def gemini_extract_from_text(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        return {}
    
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=GEMINI_EXTRACTION_PROMPT + "\n\nKullanıcı metni:\n" + t
        )
        extracted_text = resp.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(extracted_text)
    except Exception as e:
        print("Gemini API Hatası:", e)
        return {}


def map_to_features(form: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    text = (form.get("text") or "").strip()
    # Metni Regex yerine Gemini'ye gönderiyoruz
    extracted = gemini_extract_from_text(text) if text else {}

    age = safe_float(form.get("age"), None)
    if age is None:
        age = safe_float(extracted.get("age"), None)

    gender_raw = form.get("gender", "") or extracted.get("gender", "")
    mall_raw = form.get("shopping_mall", "") or extracted.get("shopping_mall", "")
    pay_raw = form.get("payment_method", "") or extracted.get("payment_method", "")

    price = safe_float(form.get("price"), None)
    if price is None:
        price = safe_float(extracted.get("price"), None)

    qty = safe_float(form.get("quantity"), None)
    if qty is None:
        qty = safe_float(extracted.get("quantity"), None)

    date_str = parse_invoice_date_or_default(form, extracted)

    if age is None or age <= 0:
        age = 30.0
    if price is None or price <= 0:
        price = 100.0
    if qty is None or qty <= 0:
        qty = 1.0

    import datetime as _dt
    yyyy = safe_int(date_str[0:4])
    mm = safe_int(date_str[5:7])
    dd = safe_int(date_str[8:10])
    wd = _dt.date(yyyy, mm, dd).weekday()
    is_weekend = 1 if wd >= 5 else 0
    
    # Gemini'den gelen day_type verisini kullanıyoruz
    day_type = extracted.get("day_type")
    if day_type == "Weekend":
        is_weekend = 1
    elif day_type == "Weekday":
        is_weekend = 0

    unit_price = price / qty
    price_log = np.log1p(price)

    q33 = float(thresholds["q33"])
    q66 = float(thresholds["q66"])
    q90 = float(thresholds["q90"])

    if price < q33:
        spend_level_encoded = 0
    elif price < q66:
        spend_level_encoded = 1
    else:
        spend_level_encoded = 2

    is_premium = 1 if price >= q90 else 0

    gender = normalize_gender(gender_raw)

    gender_enc = encoders.get("gender")
    mall_enc = encoders.get("shopping_mall")
    pay_enc = encoders.get("payment_method")

    gender_id = encoder_get_id(gender_enc, gender)
    mall_id = encoder_get_id(mall_enc, str(mall_raw).strip())
    pay_id = encoder_get_id(pay_enc, str(pay_raw).strip())

    scaled = scaler.transform([[age, price_log, unit_price]])[0]
    age_scaled, price_log_scaled, unit_price_scaled = float(scaled[0]), float(scaled[1]), float(scaled[2])

    X = np.array([[
        age_scaled,
        gender_id,
        mall_id,
        pay_id,
        is_weekend,
        price_log_scaled,
        unit_price_scaled,
        spend_level_encoded,
        is_premium
    ]], dtype=np.float32)

    debug = {
        "incoming_keys": list(form.keys()),
        "text_used": bool(text),
        "text": text[:200] + ("..." if len(text) > 200 else ""),
        "extracted_from_text": extracted,
        "final_used": {
            "age": age,
            "gender": gender,
            "shopping_mall": mall_raw,
            "payment_method": pay_raw,
            "invoice_date": date_str,
            "price": price,
            "quantity": qty
        },
        "derived": {
            "weekday": wd,
            "is_weekend": is_weekend,
            "unit_price": unit_price,
            "price_log": float(price_log),
            "spend_level_encoded": spend_level_encoded,
            "is_premium": is_premium
        },
        "encoded": {"gender_id": gender_id, "mall_id": mall_id, "pay_id": pay_id},
        "scaled": {"age_scaled": age_scaled, "price_log_scaled": price_log_scaled, "unit_price_scaled": unit_price_scaled},
        "final_shape": list(X.shape),
        "encoder_types": {
            "gender": type(gender_enc).__name__ if gender_enc is not None else None,
            "shopping_mall": type(mall_enc).__name__ if mall_enc is not None else None,
            "payment_method": type(pay_enc).__name__ if pay_enc is not None else None
        }
    }

    return X, debug


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()

        X, dbg = map_to_features(data)

        preds = model.predict(X, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_cat = CATEGORIES[pred_idx] if pred_idx < len(CATEGORIES) else str(pred_idx)
        confidence = float(preds[pred_idx])
        
        # Arayüzün beklediği formata uygun olarak explanation da ekliyoruz
        explanation = f"Bu profildeki kullanıcının {pred_cat} kategorisinden alışveriş yapması bekleniyor."

        return jsonify({
            "ok": True,
            "predicted_category": pred_cat,
            "explanation": explanation,
            "confidence": confidence,
            "extracted_json": dbg.get("extracted_from_text", {}),
            "debug": dbg
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)