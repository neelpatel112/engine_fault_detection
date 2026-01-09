# app.py - SUPER SIMPLE VERSION (100% will work)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import random
from datetime import datetime
import base64

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Engine Fault Detection System is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint - SIMULATED"""
    try:
        # Check if file was uploaded
        if 'audio' not in request.files:
            # For demo, return simulated result
            return get_simulated_result()
        
        file = request.files['audio']
        
        if file.filename == '':
            return get_simulated_result()
        
        # For now, just return simulated result
        # In real version, you'd process the file here
        return get_simulated_result()
        
    except Exception as e:
        print(f"Error: {e}")
        return get_simulated_result()

def get_simulated_result():
    """Generate realistic simulated prediction results"""
    # Randomly choose between normal and faulty
    is_faulty = random.random() > 0.6  # 40% chance of faulty
    
    if is_faulty:
        # Faulty engine: high probability
        fault_probability = random.uniform(0.7, 0.95)
        status = "FAULTY"
        color = "red"
        recommendation = "🚨 Engine fault detected! Immediate inspection recommended."
    else:
        # Normal engine: low probability
        fault_probability = random.uniform(0.05, 0.3)
        status = "NORMAL"
        color = "green"
        recommendation = "✅ Engine appears to be in normal condition."
    
    confidence = "HIGH" if fault_probability > 0.7 else "MEDIUM" if fault_probability > 0.3 else "LOW"
    
    return jsonify({
        'success': True,
        'status': status,
        'probability': fault_probability,
        'color': color,
        'recommendation': recommendation,
        'model_used': 'AI Analysis',
        'confidence': confidence,
        'timestamp': datetime.now().isoformat(),
        'note': 'Demo mode - Using simulated predictions'
    })

@app.route('/api/sample', methods=['GET'])
def get_sample():
    """Get sample analysis"""
    return get_simulated_result()

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify({
        'available_models': ['AI Engine'],
        'message': 'System is running in demonstration mode'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Engine Fault Detection System Started on port {port}")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Running in DEMO MODE")
    app.run(host='0.0.0.0', port=port, debug=False)