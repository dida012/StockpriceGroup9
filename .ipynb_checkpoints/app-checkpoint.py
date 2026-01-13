from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("best_stock_price_model.pkl")

FEATURES = ["open", "high", "low", "volume", "vwap", "change", "changePercent"]

# Home route
@app.route("/")
def home():
    return "Stock Price Prediction API is running "

# =========================
# JSON API (for Postman / frontend / mobile)
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Prepare features
        features = np.array([[data[f] for f in FEATURES]])

        prediction = model.predict(features)[0]

        return jsonify({
            "predicted_closing_price": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# HTML Form Interface
# =========================
@app.route("/predict_form", methods=["GET", "POST"])
def predict_form():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            features = np.array([[
                float(request.form[f]) for f in FEATURES
            ]])

            prediction = model.predict(features)[0]

        except Exception as e:
            error = str(e)

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Price Prediction</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            input { margin: 5px 0; padding: 6px; width: 200px; }
            button { padding: 8px 15px; margin-top: 10px; }
            .result { margin-top: 20px; font-size: 18px; color: green; }
            .error { margin-top: 20px; color: red; }
        </style>
    </head>
    <body>
        <h2>Stock Price Prediction Form </h2>

        <form method="POST">
            {% for f in features %}
                <label>{{ f }}:</label><br>
                <input type="number" step="any" name="{{ f }}" required><br>
            {% endfor %}
            <button type="submit">Predict</button>
        </form>

        {% if prediction %}
            <div class="result">
                Predicted Closing Price: {{ prediction }}
            </div>
        {% endif %}

        {% if error %}
            <div class="error">
                Error: {{ error }}
            </div>
        {% endif %}
    </body>
    </html>
    """

    return render_template_string(
        html,
        prediction=prediction,
        error=error,
        features=FEATURES
    )


# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
