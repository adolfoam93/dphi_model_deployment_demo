from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import util

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/get_data")
def get_data():
    response = jsonify({
        "genders": util.get_genders(),
        "education": util.get_education(),
        "marital_status": util.get_marital_status(),
        "dependents": util.get_dependents(),
        "self_employed": util.get_self_employed(),
        "property_area": util.get_property_area()
    })

    return response

@app.route("/predict", methods=["POST"])
def predict():
    
    ap_income = request.form['ap_income']
    coap_income = request.form['coap_income']
    loan_amt = request.form['loan_amt']
    loan_amt_term = request.form['loan_amt_term']
    credit_hist = request.form['credit_hist']
    gender = request.form['gender']
    education = request.form['education']
    married_status = request.form['married_status']
    dependents = request.form['dependents']
    self_employed = request.form['self_employed']
    property_area = request.form['property_area']

    prediction = util.get_model_prediction(ap_income, coap_income, loan_amt, loan_amt_term, credit_hist, gender, education, married_status, dependents, self_employed, property_area)

    return render_template('index.html', prediction_text=f"Loan: {prediction}")

if __name__ == "__main__":
    print("Starting Python Flask server")
    util.load_saved_artifacts()
    app.run(debug=True)