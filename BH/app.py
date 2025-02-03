from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import io
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure secret key

# Configure Gemini API key (make sure to keep it secret in production!)
genai.configure(api_key="AIzaSyAPnLDbJFxKHbjCw6E1EZSJSwB-N6gjv8I")

# ---------- Utility Functions (adapted from your script) ----------

def load_stock_data(file_stream):
    """Load and preprocess French-formatted stock data from an uploaded file stream"""
    try:
        # Read CSV with French formatting details
        df = pd.read_csv(
            file_stream,
            sep=';',         # Semicolon as delimiter
            decimal=',',     # Comma as decimal separator
            thousands=' ',   # Space as thousand separator
            parse_dates=['date'],
            dayfirst=True    # Date format: DD/MM/YYYY
        )

        # Translate French column names to English
        column_translation = {
            'symbole': 'Symbol',
            'date': 'Date',
            'ouverture': 'Open',
            'haut': 'High',
            'bas': 'Low',
            'cloture': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=column_translation)

        # Verify required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise Exception(f"Missing columns: {', '.join(missing)}")

        # Clean and sort data
        df = df.sort_values('Date').set_index('Date').dropna()
        return df

    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}\n"
                        "Common solutions:\n"
                        "1. Ensure the CSV uses semicolon (;) as column separator\n"
                        "2. Ensure decimals use a comma (,)\n"
                        "3. Date format should be DD/MM/YYYY")

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stock_data(df):
    """Perform technical analysis and generate visualizations"""
    try:
        # Calculate technical indicators
        df['50_MA'] = df['Close'].rolling(window=50).mean()
        df['200_MA'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = calculate_rsi(df['Close'])

        # Create the technical analysis chart
        plt.figure(figsize=(14, 10))

        # Price Chart with Moving Averages
        plt.subplot(3, 1, 1)
        plt.plot(df['Close'], label='Closing Price', color='blue')
        plt.plot(df['50_MA'], label='50-Day MA', color='orange')
        plt.plot(df['200_MA'], label='200-Day MA', color='green')
        plt.title('Price and Moving Averages')
        plt.legend()

        # RSI Chart
        plt.subplot(3, 1, 2)
        plt.plot(df['RSI'], label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--')
        plt.axhline(30, color='green', linestyle='--')
        plt.title('Relative Strength Index (RSI)')
        plt.ylim(0, 100)

        # Volume Chart
        plt.subplot(3, 1, 3)
        plt.bar(df.index, df['Volume'], color='gray')
        plt.title('Trading Volume')

        plt.tight_layout()

        # Save the plot to a BytesIO object
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        plt.close()
        img_bytes.seek(0)

        return df, img_bytes

    except Exception as e:
        raise Exception(f"Analysis error: {str(e)}")

def get_gemini_insights(df):
    """Generate AI-powered market analysis using the latest stock data"""
    try:
        latest = df.iloc[-1]
        summary = (
            f"Stock Analysis Summary:\n"
            f"- Latest Close: ${latest['Close']:.2f}\n"
            f"- 50-Day MA: ${latest['50_MA']:.2f} ({'Above' if latest['Close'] > latest['50_MA'] else 'Below'})\n"
            f"- 200-Day MA: ${latest['200_MA']:.2f} ({'Above' if latest['Close'] > latest['200_MA'] else 'Below'})\n"
            f"- RSI: {latest['RSI']:.1f} ({'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'})\n"
            f"- Volume: {latest['Volume']:,.0f}\n"
        )

        # Use Gemini to generate content based on the summary
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            f"Analyze this stock data and provide professional insights:\n{summary}\n"
            "Include technical analysis, trend identification, and risk assessment."
        )
        return response.text

    except Exception as e:
        raise Exception(f"AI analysis failed: {str(e)}")

# ---------- Flask Routes ----------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file was uploaded
        if 'file' not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)

        try:
            # Load and process the stock data
            df = load_stock_data(file)
            df, chart_bytes = analyze_stock_data(df)
            insights = get_gemini_insights(df)

            # Save the chart image temporarily to serve in the result template
            chart_filename = 'static/technical_analysis.png'
            with open(chart_filename, 'wb') as f:
                f.write(chart_bytes.getbuffer())

            # Render the results page with the insights and chart image
            return render_template('result.html', insights=insights, chart_url=url_for('static', filename='technical_analysis.png'))

        except Exception as e:
            flash(str(e))
            return redirect(request.url)

    # GET request - render the file upload form
    return render_template('index.html')


# Run the app
if __name__ == '__main__':
    # Create the static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
