# app.py - TensorFlow-free version
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import librosa
import joblib
import tempfile
from datetime import datetime
import json

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg'}

# Create temp directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for models
SCALER = None
RF_MODEL = None
MODEL_LOADED = False

def load_models():
    """Load ML models - ONLY Random Forest (no TensorFlow)"""
    global SCALER, RF_MODEL, MODEL_LOADED
    
    if MODEL_LOADED:
        return True
    
    try:
        print("🔄 Loading models...")
        
        # Load scaler
        SCALER = joblib.load('saved_models/scaler.pkl')
        print("✅ Scaler loaded")
        
        # Load ONLY Random Forest model (skip CNN)
        RF_MODEL = joblib.load('saved_models/random_forest_model.pkl')
        print("✅ Random Forest model loaded")
        
        MODEL_LOADED = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return False

def extract_features(file_path):
    """Extract audio features from file"""
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Chroma features
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_var = np.var(spectral_rolloff)
        
        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        
        # Combine features
        features = np.concatenate([
            mfccs_mean[:5],
            [chroma_stft_mean, chroma_stft_var],
            [rms_mean, rms_var],
            [spectral_centroid_mean, spectral_centroid_var],
            [spectral_bandwidth_mean, spectral_bandwidth_var],
            [spectral_rolloff_mean, spectral_rolloff_var],
            [zero_crossing_rate_mean, zero_crossing_rate_var],
            [tempo]
        ])
        
        return features
        
    except Exception as e:
        print(f"❌ Feature extraction error: {str(e)}")
        return None

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': MODEL_LOADED,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint - Only Random Forest"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Force use Random Forest (CNN not available)
    model_type = 'rf'
    
    if not load_models():
        return jsonify({'error': 'Models failed to load'}), 500
    
    try:
        # Save temporary file
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{datetime.now().timestamp()}.wav")
        file.save(temp_path)
        
        # Extract features
        features = extract_features(temp_path)
        if features is None:
            return jsonify({'error': 'Failed to extract audio features'}), 500
        
        # Reshape and scale
        features = features.reshape(1, -1)
        features_scaled = SCALER.transform(features)
        
        # Make prediction (only Random Forest)
        prediction = RF_MODEL.predict(features_scaled)[0]
        probability = RF_MODEL.predict_proba(features_scaled)[0]
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Prepare response
        fault_probability = float(probability[1])
        
        if prediction == 0:
            status = "NORMAL"
            color = "green"
            recommendation = "Engine appears to be in normal condition"
        else:
            status = "FAULTY"
            color = "red"
            recommendation = "Engine fault detected. Inspection recommended"
        
        return jsonify({
            'success': True,
            'status': status,
            'probability': fault_probability,
            'color': color,
            'recommendation': recommendation,
            'model_used': 'Random Forest',
            'confidence': 'HIGH' if fault_probability > 0.7 else 'MEDIUM' if fault_probability > 0.3 else 'LOW',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/models')
def get_models():
    """Get available models"""
    return jsonify({
        'available_models': ['Random Forest'],
        'cnn_available': False,
        'message': 'Using Random Forest model only (lightweight)'
    })

if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Engine Fault Detection Server on port {port}")
    print(f"📊 Using Random Forest model only (TensorFlow not available)")
    app.run(host='0.0.0.0', port=port, debug=False)