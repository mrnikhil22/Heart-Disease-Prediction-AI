from flask import Flask, request, render_template, send_file
from tensorflow.keras.models import load_model
import numpy as np
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
import datetime
import os
import joblib   # ✅ ADDED

app = Flask(__name__)

# Load model
model = load_model('mymodel.keras')

# ✅ LOAD SCALER (IMPORTANT)
scaler = joblib.load("scaler.pkl")


# Risk calculation
def compute_risk_and_result(prediction_array):
    prob = float(prediction_array[0][0])
    prob = max(0.0, min(1.0, prob))

    risk_score = round(prob * 100, 1)

    if prob >= 0.75:
        category = "High Risk"
    elif prob >= 0.4:
        category = "Medium Risk"
    else:
        category = "Low Risk"

    return risk_score, category, prob


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['chestpain']),
            float(request.form['restingBP']),
            float(request.form['serumcholestrol']),
            float(request.form['fastingbloodsugar']),
            float(request.form['restingrelectro']),
            float(request.form['maxheartrate']),
            float(request.form['exerciseangia']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['noofmajorvessels']),
            float(request.form['thal'])
        ]

        input_array = np.array([inputs])

            # ✅ FIX: Ensure correct column order (VERY IMPORTANT)
        input_array = input_array.reshape(1, -1)

           # ✅ APPLY SCALING
        input_array = scaler.transform(input_array)

        

        prediction = model.predict(input_array)

        risk_score, category, prob = compute_risk_and_result(prediction)

        return render_template(
            'result.html',
            risk_score=risk_score,
            category=category,
            probability=round(prob, 3)
        )

    except Exception as e:
        print("ERROR:", e)
        return "<h2>Invalid Input! Please enter correct values.</h2>"


@app.route('/report', methods=['POST'])
def report():
    try:
        risk_score = request.form['risk_score']
        category = request.form['category']
        probability = request.form['probability']

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            name='TitleStyle',
            fontSize=20,
            textColor=colors.red,
            spaceAfter=20,
            alignment=1
        )

        normal_style = styles["Normal"]

        content = []

        content.append(Paragraph("❤️ Heart Disease Risk Report", title_style))
        content.append(Spacer(1, 15))

        date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
        content.append(Paragraph(f"<b>Date:</b> {date}", normal_style))
        content.append(Spacer(1, 15))

        data = [
            ["Parameter", "Value"],
            ["Risk Level", category],
            ["Risk Score", f"{risk_score}%"],
            ["Probability", probability]
        ]

        table = Table(data, colWidths=[200, 200])

        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.red),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('GRID', (0,0),(-1,-1),1,colors.black),
        ]))

        content.append(table)

        doc.build(content)

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="Heart_Report.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        return str(e)


# ✅ IMPORTANT FOR RENDER
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))   # ⚠️ 5000 mat use karna
    app.run(host='0.0.0.0', port=port)