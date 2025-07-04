from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import os
from catboost import CatBoostClassifier
from urllib.parse import quote
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = 'your-secret-key'
ALLOWED_EXTENSIONS = {'csv'}

# Load your trained CatBoost model
import joblib

model_path = 'model/final_catboost_model.pkl'
model = joblib.load(model_path)


# Uppercase SELECTED_GENES once for consistent matching
SELECTED_GENES = [g.upper() for g in [
    'DPF1', 'HSD17B6', 'SH2D2A', 'C6', 'HOMER3', 'ORC6', 'IL11', 'CRISP3', 
    'MMP11', 'CAB39L', 'CD276', 'CCN4', 'CCNE1', 'COL1A1', 'STC2', 'HJURP', 
    'HOXD11', 'C1QTNF6', 'LOXL2', 'CEP55', 'P3H4', 'CDCA5', 'ADAM12', 
    'SERPINH1', 'LPCAT1', 'NRG2', 'CAMK2N2', 'ESM1', 'CTHRC1', 'AQP7', 
    'PLIN1', 'GPD1', 'BMP1', 'GPRIN1', 'KRT4', 'LBX2', 'HOXC9', 'ADIPOQ', 
    'TMEM132C', 'CIDEC', 'COL13A1', 'HOXC6', 'ADH4', 'FADS3', 'SLC26A6', 
    'FOXD2-AS1', 'LINC02156', 'LINCADL'
]]

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_cancer(file_path):
    try:
        df = pd.read_csv(file_path)

        # Normalize gene symbols to uppercase for matching
        df['gene_symbol'] = df['gene_symbol'].str.upper()

        # Filter genes to selected set
        df = df[df['gene_symbol'].isin(SELECTED_GENES)]

        if df.empty:
            return {"error": "None of the required genes were found in the file.", "status": "error"}

        # Pivot to get gene expression vector in correct order, fill missing with 0
        gene_vector = df.set_index('gene_symbol')['expression_value'].reindex(SELECTED_GENES).fillna(0)

        X_input = gene_vector.values.reshape(1, -1)

        # Predict and get confidence
        prediction = model.predict(X_input)[0]
        confidence = float(model.predict_proba(X_input)[0][1])

        # Get top 10 important genes from model feature importance
        importances = model.get_feature_importance()
        top_indices = np.argsort(importances)[-10:][::-1]
        top_genes = [SELECTED_GENES[i] for i in top_indices]
        top_importance = [importances[i] for i in top_indices]

        # Store in session for visualization
        session['importance_labels'] = top_genes
        session['importance_values'] = top_importance
        session['expression_values'] = gene_vector.tolist()  # Store expression values for visualization

        return {
            "prediction": "Cancerous" if prediction == 1 else "Non-Cancerous",
            "confidence": round(confidence, 3),
            "importance_labels": top_genes,
            "importance_values": top_importance,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded", "status": "error"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected", "status": "error"})

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result = predict_cancer(file_path)

            if result["status"] == "success":
                session['prediction'] = result["prediction"]
                session['confidence'] = result["confidence"]
            os.remove(file_path)
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e), "status": "error"})

    return jsonify({"error": "Invalid file type", "status": "error"})

@app.route('/glossary')
def glossary():
    return render_template('glossary.html')



@app.route('/visualizations')
def visualizations():
    # Get stored values from session with defaults
    labels = session.get('importance_labels', [])
    values = session.get('importance_values', [])
    expression_values = session.get('expression_values', [])
    
    # If no data is available, initialize with empty lists
    if not labels or not values or not expression_values:
        return render_template('visualizations.html', 
                            labels=[],
                            values=[],
                            expression_values=[])
    
    return render_template('visualizations.html', 
                         labels=labels,
                         values=values,
                         expression_values=expression_values)

if __name__ == '__main__':
    app.run(debug=True)
