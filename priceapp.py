from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO
import seaborn as sns

# Use non-interactive backend for matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load your saved model
model = joblib.load("best_stock_price_model.pkl")

# Try to load the cleaned dataset for visualizations
try:
    df = pd.read_csv("clean_stock_data.csv")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
except:
    df = None

# Helper function to convert plot to base64
def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url

# Home page
@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Price Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .nav-buttons { text-align: center; margin: 20px 0; }
            .nav-buttons a { display: inline-block; margin: 10px; padding: 12px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }
            .nav-buttons a:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Stock Price Prediction System</h1>
            <div class="nav-buttons">
                <a href="/predict_form">Make Prediction</a>
                <a href="/visualizations">View Model Visualizations</a>
                <a href="/model_comparison">Model Comparison</a>
            </div>
        </div>
    </body>
    </html>
    """)

# Generate model comparison chart
@app.route("/model_comparison")
def model_comparison():
    try:
        # Create evaluation data (you'll need to load actual test data or use saved metrics)
        evaluation_data = {
            "Model": ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor"],
            "RMSE": [5.2, 3.8, 4.5],  # Replace with actual values from your training
            "R2 Score": [0.92, 0.96, 0.94]  # Replace with actual values from your training
        }
        
        evaluation_df = pd.DataFrame(evaluation_data)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # RMSE Comparison
        rmse_values = evaluation_df['RMSE'].values
        models_list = evaluation_df['Model'].values
        
        axes[0].bar(models_list, rmse_values, color=['blue', 'green', 'orange'], alpha=0.7)
        axes[0].set_ylabel('RMSE', fontsize=12)
        axes[0].set_title('Model Comparison: RMSE (Lower is Better)', fontsize=12, fontweight='bold')
        axes[0].set_ylim([0, max(rmse_values) * 1.2])
        for i, v in enumerate(rmse_values):
            axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R² Score Comparison
        r2_values = evaluation_df['R2 Score'].values
        
        axes[1].bar(models_list, r2_values, color=['blue', 'green', 'orange'], alpha=0.7)
        axes[1].set_ylabel('R² Score', fontsize=12)
        axes[1].set_title('Model Comparison: R² Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 1])
        for i, v in enumerate(r2_values):
            axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .chart { text-align: center; margin: 20px 0; }
                img { max-width: 100%; height: auto; }
                .back-btn { display: inline-block; margin: 20px 0; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }
                .back-btn:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Model Performance Comparison</h1>
                <div class="chart">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Model Comparison Chart">
                </div>
                <a href="/" class="back-btn">← Back to Home</a>
            </div>
        </body>
        </html>
        """, plot_url=plot_url)
    except Exception as e:
        return f"Error generating chart: {e}"

# Visualizations page
@app.route("/visualizations")
def visualizations():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .visualization { margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 5px; }
            .visualization h2 { color: #007bff; }
            .visualization-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
            .viz-card { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .viz-card a { display: inline-block; padding: 10px 20px; background-color: #28a745; color: white; text-decoration: none; border-radius: 5px; margin-top: 10px; }
            .viz-card a:hover { background-color: #218838; }
            .back-btn { display: inline-block; margin: 20px 0; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }
            .back-btn:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Model Visualizations & Analysis</h1>
            
            <div class="visualization">
                <h2>📊 All Available Visualizations</h2>
                <div class="visualization-grid">
                    <div class="viz-card">
                        <h3>🎯 Model Comparison</h3>
                        <p>RMSE and R² Score comparison across all three models</p>
                        <a href="/model_comparison">View Chart</a>
                    </div>
                    <div class="viz-card">
                        <h3>📈 Time Series</h3>
                        <p>Historical closing price movement over time</p>
                        <a href="/time_series_plot">View Chart</a>
                    </div>
                    <div class="viz-card">
                        <h3>📊 Price Distribution</h3>
                        <p>Distribution of closing prices with KDE</p>
                        <a href="/price_distribution">View Chart</a>
                    </div>
                    <div class="viz-card">
                        <h3>🔗 Correlation Heatmap</h3>
                        <p>Feature correlations with closing price</p>
                        <a href="/correlation_heatmap">View Chart</a>
                    </div>
                    <div class="viz-card">
                        <h3>💹 Feature Scatter Plots</h3>
                        <p>Individual features vs closing price</p>
                        <a href="/feature_scatter">View Charts</a>
                    </div>
                    <div class="viz-card">
                        <h3>✅ Actual vs Predicted</h3>
                        <p>Model predictions compared to actual values</p>
                        <a href="/actual_vs_predicted">View Chart</a>
                    </div>
                    <div class="viz-card">
                        <h3>📉 Residuals Analysis</h3>
                        <p>Model error distribution</p>
                        <a href="/residuals_plot">View Chart</a>
                    </div>
                </div>
            </div>
            
            <a href="/" class="back-btn">← Back to Home</a>
        </div>
    </body>
    </html>
    """)



# Time Series Plot
@app.route("/time_series_plot")
def time_series_plot():
    if df is None:
        return "Dataset not loaded"
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['date'], df['close'], color='blue', linewidth=2)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Closing Price", fontsize=12)
        ax.set_title("Time Series of Closing Price", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plot_url = fig_to_base64(fig)
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Time Series Plot</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .chart { text-align: center; margin: 20px 0; }
                img { max-width: 100%; height: auto; }
                .description { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .back-btn { display: inline-block; margin: 20px 0; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }
                .back-btn:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 Time Series of Closing Price</h1>
                <div class="description">
                    <p><strong>What this shows:</strong> Historical movement of the stock's closing price over time, including trends, volatility, and market behavior.</p>
                    <p><strong>Why it matters:</strong> Understanding historical patterns helps justify prediction models and identify temporal trends.</p>
                </div>
                <div class="chart">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Time Series Plot">
                </div>
                <a href="/visualizations" class="back-btn">← Back to Visualizations</a>
            </div>
        </body>
        </html>
        """, plot_url=plot_url)
    except Exception as e:
        return f"Error: {e}"

# Price Distribution
@app.route("/price_distribution")
def price_distribution():
    if df is None:
        return "Dataset not loaded"
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['close'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel("Closing Price", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Closing Prices", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plot_url = fig_to_base64(fig)
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Price Distribution</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .chart { text-align: center; margin: 20px 0; }
                img { max-width: 100%; height: auto; }
                .description { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .back-btn { display: inline-block; margin: 20px 0; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }
                .back-btn:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Distribution of Closing Prices</h1>
                <div class="description">
                    <p><strong>What this shows:</strong> The range, spread, and frequency of closing prices across trading days.</p>
                    <p><strong>Why it matters:</strong> Identifies skewness, outliers, and variability that could affect model performance.</p>
                </div>
                <div class="chart">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Price Distribution">
                </div>
                <a href="/visualizations" class="back-btn">← Back to Visualizations</a>
            </div>
        </body>
        </html>
        """, plot_url=plot_url)
    except Exception as e:
        return f"Error: {e}"

# Correlation Heatmap
@app.route("/correlation_heatmap")
def correlation_heatmap():
    if df is None:
        return "Dataset not loaded"
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[['open', 'high', 'low', 'close', 'volume', 'vwap', 'change', 'changePercent']].corr()
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title("Correlation Matrix of Stock Features", fontsize=14, fontweight='bold')
        
        plot_url = fig_to_base64(fig)
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Correlation Heatmap</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .chart { text-align: center; margin: 20px 0; }
                img { max-width: 100%; height: auto; }
                .description { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .back-btn { display: inline-block; margin: 20px 0; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }
                .back-btn:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔗 Correlation Heatmap</h1>
                <div class="description">
                    <p><strong>What this shows:</strong> Correlation coefficients between all features and the target variable (close price).</p>
                    <p><strong>Why it matters:</strong> Identifies the strongest predictors of closing price and detects multicollinearity between features.</p>
                    <p><strong>How to read it:</strong> Values range from -1 (negative correlation) to +1 (positive correlation). Dark blue = strong correlation.</p>
                </div>
                <div class="chart">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Correlation Heatmap">
                </div>
                <a href="/visualizations" class="back-btn">← Back to Visualizations</a>
            </div>
        </body>
        </html>
        """, plot_url=plot_url)
    except Exception as e:
        return f"Error: {e}"

# Feature Scatter Plots
@app.route("/feature_scatter")
def feature_scatter():
    if df is None:
        return "Dataset not loaded"
    try:
        features = ['open', 'high', 'low', 'vwap', 'volume']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            axes[idx].scatter(df[feature], df['close'], alpha=0.5, s=20)
            axes[idx].set_xlabel(feature.capitalize(), fontsize=10)
            axes[idx].set_ylabel("Closing Price", fontsize=10)
            axes[idx].set_title(f"{feature.capitalize()} vs Closing Price", fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].axis('off')
        plt.tight_layout()
        
        plot_url = fig_to_base64(fig)
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Scatter Plots</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .chart { text-align: center; margin: 20px 0; }
                img { max-width: 100%; height: auto; }
                .description { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .back-btn { display: inline-block; margin: 20px 0; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }
                .back-btn:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>💹 Feature vs Closing Price Scatter Plots</h1>
                <div class="description">
                    <p><strong>What this shows:</strong> Individual relationships between each feature and the target closing price.</p>
                    <p><strong>Why it matters:</strong> Reveals linear and non-linear patterns that justify feature selection for the model.</p>
                </div>
                <div class="chart">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Feature Scatter Plots">
                </div>
                <a href="/visualizations" class="back-btn">← Back to Visualizations</a>
            </div>
        </body>
        </html>
        """, plot_url=plot_url)
    except Exception as e:
        return f"Error: {e}"

# Actual vs Predicted
@app.route("/actual_vs_predicted")
def actual_vs_predicted():
    if df is None:
        return "Dataset not loaded"
    try:
        from sklearn.model_selection import train_test_split
        features = ["open", "high", "low", "volume", "vwap", "change", "changePercent"]
        X = df[features]
        y = df["close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, y_pred, alpha=0.5, s=30)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel("Actual Closing Price", fontsize=12)
        ax.set_ylabel("Predicted Closing Price", fontsize=12)
        ax.set_title("Actual vs Predicted Closing Price", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_url = fig_to_base64(fig)
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Actual vs Predicted</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .chart { text-align: center; margin: 20px 0; }
                img { max-width: 100%; height: auto; }
                .description { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .back-btn { display: inline-block; margin: 20px 0; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }
                .back-btn:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>✅ Actual vs Predicted Closing Price</h1>
                <div class="description">
                    <p><strong>What this shows:</strong> Comparison of actual test values against model predictions.</p>
                    <p><strong>Why it matters:</strong> Points close to the red diagonal line indicate high prediction accuracy.</p>
                    <p><strong>How to read it:</strong> The diagonal red line represents perfect predictions. Clustering around this line = good model.</p>
                </div>
                <div class="chart">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Actual vs Predicted">
                </div>
                <a href="/visualizations" class="back-btn">← Back to Visualizations</a>
            </div>
        </body>
        </html>
        """, plot_url=plot_url)
    except Exception as e:
        return f"Error: {e}"

# Residuals Plot
@app.route("/residuals_plot")
def residuals_plot():
    if df is None:
        return "Dataset not loaded"
    try:
        from sklearn.model_selection import train_test_split
        features = ["open", "high", "low", "volume", "vwap", "change", "changePercent"]
        X = df[features]
        y = df["close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax.axhline(0, linestyle='--', color='red', lw=2)
        ax.set_xlabel("Predicted Closing Price", fontsize=12)
        ax.set_ylabel("Residuals", fontsize=12)
        ax.set_title("Residuals vs Predicted Values", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plot_url = fig_to_base64(fig)
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Residuals Plot</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .chart { text-align: center; margin: 20px 0; }
                img { max-width: 100%; height: auto; }
                .description { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .back-btn { display: inline-block; margin: 20px 0; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }
                .back-btn:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📉 Residuals Analysis</h1>
                <div class="description">
                    <p><strong>What this shows:</strong> The distribution of prediction errors (residuals) across predicted values.</p>
                    <p><strong>Why it matters:</strong> Random residuals around zero indicate a good model. Patterns suggest model bias.</p>
                    <p><strong>How to read it:</strong> Points should be randomly scattered around the red line with no clear pattern.</p>
                </div>
                <div class="chart">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Residuals Plot">
                </div>
                <a href="/visualizations" class="back-btn">← Back to Visualizations</a>
            </div>
        </body>
        </html>
        """, plot_url=plot_url)
    except Exception as e:
        return f"Error: {e}"


@app.route("/predict_form", methods=["GET", "POST"])
def predict_form():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            # Read input values from form
            features = np.array([[float(request.form[f"f{i}"]) for i in range(1, 8)]])
            prediction = float(model.predict(features)[0])
        except ValueError:
            error = "Error: Please enter valid numeric values for all features"
        except Exception as e:
            error = f"Error: {e}"

    # HTML template
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Price Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; }
            .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); max-width: 500px; width: 100%; }
            h2 { color: #333; text-align: center; margin-bottom: 30px; }
            .form-group { margin: 15px 0; }
            label { display: block; color: #555; font-weight: bold; margin-bottom: 5px; }
            input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; font-size: 14px; }
            input:focus { outline: none; border-color: #667eea; box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
            button { width: 100%; padding: 12px; background-color: #667eea; color: white; border: none; border-radius: 5px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 20px; }
            button:hover { background-color: #764ba2; }
            .result { margin-top: 20px; padding: 15px; background-color: #d4edda; border-left: 4px solid #28a745; border-radius: 5px; font-weight: bold; color: #155724; }
            .error { margin-top: 20px; padding: 15px; background-color: #f8d7da; border-left: 4px solid #dc3545; border-radius: 5px; font-weight: bold; color: #721c24; }
            .feature-info { font-size: 12px; color: #888; margin-top: 2px; }
            .back-btn { display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #6c757d; color: white; text-decoration: none; border-radius: 5px; text-align: center; }
            .back-btn:hover { background-color: #5a6268; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚀 Stock Price Prediction</h2>
            <form method="POST">
                <div class="form-group">
                    <label>1. Opening Price:</label>
                    <input type="number" step="0.01" name="f1" required placeholder="e.g., 100.50">
                    <div class="feature-info">Current opening price of the stock</div>
                </div>
                <div class="form-group">
                    <label>2. Highest Price:</label>
                    <input type="number" step="0.01" name="f2" required placeholder="e.g., 105.75">
                    <div class="feature-info">Highest price during trading period</div>
                </div>
                <div class="form-group">
                    <label>3. Lowest Price:</label>
                    <input type="number" step="0.01" name="f3" required placeholder="e.g., 98.50">
                    <div class="feature-info">Lowest price during trading period</div>
                </div>
                <div class="form-group">
                    <label>4. Trading Volume:</label>
                    <input type="number" step="1" name="f4" required placeholder="e.g., 1000000">
                    <div class="feature-info">Total shares traded</div>
                </div>
                <div class="form-group">
                    <label>5. Volume Weighted Avg Price:</label>
                    <input type="number" step="0.01" name="f5" required placeholder="e.g., 101.25">
                    <div class="feature-info">VWAP indicator</div>
                </div>
                <div class="form-group">
                    <label>6. Price Change:</label>
                    <input type="number" step="0.01" name="f6" required placeholder="e.g., 1.50">
                    <div class="feature-info">Absolute change from previous close</div>
                </div>
                <div class="form-group">
                    <label>7. Percent Change:</label>
                    <input type="number" step="0.01" name="f7" required placeholder="e.g., 1.5">
                    <div class="feature-info">Percentage change from previous close</div>
                </div>
                
                <button type="submit">🔮 Predict Stock Price</button>
            </form>
            
            {% if error %}
            <div class="error">{{ error }}</div>
            {% endif %}
            
            {% if prediction is not none %}
            <div class="result">
                📈 Predicted Closing Price: ${{ "%.2f"|format(prediction) }}
            </div>
            {% endif %}
            
            <a href="/" class="back-btn">← Back to Home</a>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
