from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# =========================
# Load model & dataset
# =========================
model = joblib.load("best_stock_price_model.pkl")
#df = pd.read_csv("stock_data.csv")

FEATURES = ["open", "high", "low", "volume", "vwap", "change", "changePercent"]
TARGET = "close"


# =========================
# Helper: Convert plot to image
# =========================
def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =========================
# Home
# =========================
@app.route("/")
def home():
    return """
    <h2>Stock Price Prediction API</h2>
    <ul>
        <li>/predict (POST JSON)</li>
        <li>/predict_form</li>
        <li>/correlation</li>
        <li>/feature_importance</li>
        <li>/actual_vs_predicted</li>
        <li>/residuals</li>
    </ul>
    """


# =========================
# Prediction API (JSON)
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        X = np.array([[data[f] for f in FEATURES]])
        prediction = model.predict(X)[0]

        return jsonify({"predicted_closing_price": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# HTML Prediction Form
# =========================
@app.route("/predict_form", methods=["GET", "POST"])
def predict_form():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            X = np.array([[float(request.form[f]) for f in FEATURES]])
            prediction = model.predict(X)[0]
        except Exception as e:
            error = str(e)

    html = """
    <html>
    <head>
        <title>Stock Prediction</title>
    </head>
    <body>
        <h2>Stock Price Prediction</h2>
        <form method="POST">
            {% for f in features %}
                <label>{{ f }}</label><br>
                <input type="number" step="any" name="{{ f }}" required><br><br>
            {% endfor %}
            <button type="submit">Predict</button>
        </form>

        {% if prediction %}
            <h3>Predicted Closing Price: {{ prediction }}</h3>
        {% endif %}

        {% if error %}
            <p style="color:red;">{{ error }}</p>
        {% endif %}
    </body>
    </html>
    """

    return render_template_string(html, features=FEATURES,
                                  prediction=prediction, error=error)


# =========================
# Correlation Heatmap
# =========================
@app.route("/correlation")
def correlation():
    corr = df[FEATURES + [TARGET]].corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")

    img = plot_to_base64()
    return f"<h2>Correlation Heatmap</h2><img src='data:image/png;base64,{img}'>"


# =========================
# Feature Importance
# =========================
@app.route("/feature_importance")
def feature_importance():
    importance = model.coef_

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance, y=FEATURES)
    plt.title("Feature Importance")

    img = plot_to_base64()
    return f"<h2>Feature Importance</h2><img src='data:image/png;base64,{img}'>"


# =========================
# Actual vs Predicted
# =========================
@app.route("/actual_vs_predicted")
def actual_vs_predicted():
    X = df[FEATURES]
    y = df[TARGET]
    preds = model.predict(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(y, preds, alpha=0.5)
    plt.xlabel("Actual Close")
    plt.ylabel("Predicted Close")
    plt.title("Actual vs Predicted")

    img = plot_to_base64()
    return f"<h2>Actual vs Predicted</h2><img src='data:image/png;base64,{img}'>"


# =========================
# Residuals Plot
# =========================
@app.route("/residuals")
def residuals():
    X = df[FEATURES]
    y = df[TARGET]
    preds = model.predict(X)
    residuals = y - preds

    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")

    img = plot_to_base64()
    return f"<h2>Residuals</h2><img src='data:image/png;base64,{img}'>"


# =========================
# Run Server
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
