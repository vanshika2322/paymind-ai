from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
app = Flask(__name__)

# Load ML model
model = joblib.load("model/payment_model.pkl")
encoders = joblib.load("model/encoders.pkl")


# 🔁 Decision Engine
def get_retry_method(payment_method, bank_status, network_quality):
    if bank_status == "Down":
        return "UPI (fastest alternative)"
    elif network_quality == "Weak":
        return "Wallet (low network dependency)"
    elif payment_method == "Card":
        return "UPI (higher success rate)"
    return payment_method


# 🤖 AI Explanation (Clean + Human readable)
def get_ai_explanation(amount, payment_method, bank_status, network_quality, risk, retry_method):

    if risk == "High":
        return f"""
        This payment has a high chance of failure due to {bank_status.lower()} bank condition and {network_quality.lower()} network quality.
        It is recommended to switch to {retry_method} for better success.
        Please retry after a few seconds or use a more stable payment method.
        """

    else:
        return f"""
        This payment is likely to succeed because the bank status is {bank_status.lower()} and the network is {network_quality.lower()}.
        However, using {retry_method} can further improve reliability.
        You can proceed with confidence.
        """


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        amount = int(request.form["amount"])
        payment_method = request.form["payment_method"]
        bank_status = request.form["bank_status"]
        network_quality = request.form["network_quality"]
        hour = int(request.form["hour"])
        previous_failures = int(request.form["previous_failures"])

        data = {
            "amount": amount,
            "payment_method": encoders["payment_method"].transform([payment_method])[0],
            "bank_status": encoders["bank_status"].transform([bank_status])[0],
            "network_quality": encoders["network_quality"].transform([network_quality])[0],
            "hour": hour,
            "previous_failures": previous_failures
        }

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        risk = "Low" if prediction == 1 else "High"
        retry_method = get_retry_method(payment_method, bank_status, network_quality)

        explanation = get_ai_explanation(
            amount, payment_method, bank_status, network_quality, risk, retry_method
        )

        result = {
            "prediction": "Payment Success Likely" if prediction == 1 else "Payment Failure Likely",
            "success_probability": round(probability * 100, 2),
            "risk": risk,
            "retry_method": retry_method,
            "explanation": explanation
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)