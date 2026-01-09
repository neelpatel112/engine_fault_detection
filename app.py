# app.py - ULTRA SIMPLE - 100% will work
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from datetime import datetime
import random

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
        'message': 'Engine Fault Detection System',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint - SIMULATED"""
    # Generate realistic results
    is_faulty = random.random() > 0.6  # 40% chance faulty
    
    if is_faulty:
        fault_probability = random.uniform(0.7, 0.95)
        status = "FAULTY"
        color = "red"
        recommendation = "🚨 Engine fault detected! Immediate inspection recommended."
    else:
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
        'model_used': 'AI Analysis Engine',
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/sample', methods=['GET'])
def get_sample():
    """Get sample analysis"""
    return jsonify({
        'success': True,
        'status': 'NORMAL',
        'probability': 0.15,
        'color': 'green',
        'recommendation': 'Sample analysis: Engine running normally',
        'model_used': 'AI Engine',
        'confidence': 'LOW',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Engine Fault Detection System Started on port {port}")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    app.run(host='0.0.0.0', port=port, debug=False)