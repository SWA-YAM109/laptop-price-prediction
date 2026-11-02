import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

# Define directory paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'laptop_price_predictor.pickle')

# Fallback path if model is located in workspace root 'model' directory
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'model', 'laptop_price_predictor.pickle')

# Supported categorical options matching model training features
ALLOWED_COMPANIES = ['acer', 'apple', 'asus', 'dell', 'hp', 'lenovo', 'msi', 'other', 'toshiba']
ALLOWED_TYPENAMES = ['2in1convertible', 'gaming', 'netbook', 'notebook', 'ultrabook', 'workstation']
ALLOWED_OPSYS = ['linux', 'mac', 'other', 'windows']
ALLOWED_CPUS = ['amd', 'intelcorei3', 'intelcorei5', 'intelcorei7', 'other']
ALLOWED_GPUS = ['amd', 'intel', 'nvidia']

def load_model():
    """Safely load the trained RandomForestRegressor model and ensure compatibility."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as file:
        loaded_model = pickle.load(file)
    
    # Forward compatibility for newer scikit-learn versions
    if not hasattr(loaded_model, 'monotonic_cst'):
        loaded_model.monotonic_cst = None
    for est in getattr(loaded_model, 'estimators_', []):
        if not hasattr(est, 'monotonic_cst'):
            est.monotonic_cst = None
    return loaded_model

# Initialize model at startup
model = None
try:
    model = load_model()
except Exception as e:
    print(f"Warning: Failed to load model at startup: {e}")

# Initialize Flask application with explicit template and static paths
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

def build_feature_vector(ram, weight, touchscreen, ips, company, typename, opsys, cpu, gpu):
    """
    Constructs the 31-feature vector in the exact order required by the trained model:
    [Ram, Weight, Touchscreen, Ips,
     Company (9), TypeName (6), OpSys (4), cpu_name (5), gpu_name (3)]
    """
    features = [
        int(ram),
        float(weight),
        1 if touchscreen else 0,
        1 if ips else 0
    ]

    # One-hot encoding for Company (9 classes)
    for c in ALLOWED_COMPANIES:
        features.append(1 if company == c else 0)

    # One-hot encoding for TypeName (6 classes)
    for t in ALLOWED_TYPENAMES:
        features.append(1 if typename == t else 0)

    # One-hot encoding for OpSys (4 classes)
    for o in ALLOWED_OPSYS:
        features.append(1 if opsys == o else 0)

    # One-hot encoding for CPU (5 classes)
    for cp in ALLOWED_CPUS:
        features.append(1 if cpu == cp else 0)

    # One-hot encoding for GPU (3 classes)
    for g in ALLOWED_GPUS:
        features.append(1 if gpu == g else 0)

    return features

@app.route('/', methods=['GET', 'POST'])
def index():
    price_prediction = None
    price_formatted = None
    error_message = None

    # Default form values for clean initial presentation
    form_data = {
        'ram': '8',
        'weight': '1.80',
        'company': 'hp',
        'typename': 'notebook',
        'opsys': 'windows',
        'cpuname': 'intelcorei5',
        'gpuname': 'intel',
        'touchscreen': False,
        'ips': True
    }

    if request.method == 'POST':
        # Retrieve form data
        raw_ram = request.form.get('ram', '').strip()
        raw_weight = request.form.get('weight', '').strip()
        company = request.form.get('company', '').strip().lower()
        typename = request.form.get('typename', '').strip().lower()
        opsys = request.form.get('opsys', '').strip().lower()
        cpuname = request.form.get('cpuname', '').strip().lower()
        gpuname = request.form.get('gpuname', '').strip().lower()
        touchscreen = 'touchscreen' in request.form
        ips = 'ips' in request.form

        # Update form_data to keep user selections populated
        form_data = {
            'ram': raw_ram,
            'weight': raw_weight,
            'company': company,
            'typename': typename,
            'opsys': opsys,
            'cpuname': cpuname,
            'gpuname': gpuname,
            'touchscreen': touchscreen,
            'ips': ips
        }

        # Server-side validation
        validation_errors = []

        # Validate RAM
        try:
            ram_val = int(raw_ram)
            if ram_val < 2 or ram_val > 128:
                validation_errors.append("RAM must be between 2 GB and 128 GB.")
        except (ValueError, TypeError):
            validation_errors.append("RAM must be a valid integer number (e.g. 8, 16).")
            ram_val = None

        # Validate Weight
        try:
            weight_val = float(raw_weight)
            if weight_val < 0.5 or weight_val > 10.0:
                validation_errors.append("Weight must be between 0.5 kg and 10.0 kg.")
        except (ValueError, TypeError):
            validation_errors.append("Weight must be a valid number (e.g. 1.8).")
            weight_val = None

        # Validate Categoricals
        if company not in ALLOWED_COMPANIES:
            validation_errors.append("Please select a valid laptop brand.")
        if typename not in ALLOWED_TYPENAMES:
            validation_errors.append("Please select a valid laptop category.")
        if opsys not in ALLOWED_OPSYS:
            validation_errors.append("Please select a valid operating system.")
        if cpuname not in ALLOWED_CPUS:
            validation_errors.append("Please select a valid processor (CPU).")
        if gpuname not in ALLOWED_GPUS:
            validation_errors.append("Please select a valid graphics processor (GPU).")

        if validation_errors:
            error_message = " ".join(validation_errors)
        else:
            try:
                global model
                if model is None:
                    model = load_model()

                feature_vector = build_feature_vector(
                    ram=ram_val,
                    weight=weight_val,
                    touchscreen=touchscreen,
                    ips=ips,
                    company=company,
                    typename=typename,
                    opsys=opsys,
                    cpu=cpuname,
                    gpu=gpuname
                )

                # Predict using feature names if available to avoid scikit-learn warnings
                if hasattr(model, 'feature_names_in_'):
                    df_features = pd.DataFrame([feature_vector], columns=model.feature_names_in_)
                    pred = model.predict(df_features)
                else:
                    pred = model.predict([feature_vector])

                predicted_val = max(0.0, float(pred[0]))
                price_prediction = round(predicted_val, 2)
                price_formatted = f"{price_prediction:,.2f}"

            except Exception as e:
                # Log internally and display safe user message without exposing internal paths
                print(f"Prediction error: {e}")
                error_message = "An error occurred while generating the price estimate. Please verify inputs and try again."

    return render_template(
        "index.html",
        price_prediction=price_prediction,
        price_formatted=price_formatted,
        form_data=form_data,
        error_message=error_message
    )

@app.route('/reset', methods=['GET'])
def reset():
    """Clear form selections and return to initial state."""
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("Starting Laptop Price Predictor server on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
