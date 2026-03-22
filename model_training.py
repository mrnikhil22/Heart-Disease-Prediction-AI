from flask import Flask, request, render_template, send_file
from tensorflow.keras.models import load_model
import numpy as np
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# Load model
model = load_model('mymodel.keras')


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
            float(request.form['restingrelectro']),
            float(request.form['maxheartrate']),
            float(request.form['exerciseangia']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['noofmajorvessels']),
            float(request.form['thal'])
        ]

        # ✅ VALIDATION ADDED
        age, gender, cp, bp, chol, ecg, hr, exang, oldpeak, slope, vessels, thal = inputs

        if age <= 0 or age > 120:
            return "<h3>Invalid Age</h3>"

        if bp <= 0 or bp > 300:
            return "<h3>Invalid Blood Pressure</h3>"

        if chol <= 0 or chol > 600:
            return "<h3>Invalid Cholesterol</h3>"

        if hr <= 0 or hr > 250:
            return "<h3>Invalid Heart Rate</h3>"

        if oldpeak < 0 or oldpeak > 10:
            return "<h3>Invalid Oldpeak value</h3>"

        input_array = np.array([inputs])

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


# ----------- PDF CODE SAME -----------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
import datetime

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

        heading_style = ParagraphStyle(
            name='Heading',
            fontSize=14,
            spaceAfter=10
        )

        normal_style = styles["Normal"]

        content = []

        content.append(Paragraph("❤️ Heart Disease Risk Report", title_style))
        content.append(Paragraph("AI Health Diagnostics System", normal_style))
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
        content.append(Spacer(1, 25))

        if category == "High Risk":
            advice = "⚠️ High probability of heart disease. Immediate medical consultation is strongly recommended."
        elif category == "Medium Risk":
            advice = "⚠️ Moderate risk detected. Lifestyle changes and periodic checkups advised."
        else:
            advice = "✅ Low risk. Maintain a healthy lifestyle."

        content.append(Paragraph("<b>Medical Interpretation:</b>", heading_style))
        content.append(Paragraph(advice, normal_style))

        content.append(Spacer(1, 25))
        content.append(Paragraph("---- End of Report ----", normal_style))
        content.append(Paragraph("Generated by AI Health System", normal_style))

        doc.build(content)

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="Heart_Report_Professional.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        return str(e)


# ✅ RENDER FIX (IMPORTANT)
import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)