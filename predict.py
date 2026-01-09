
import numpy as np
import librosa
import joblib
import tensorflow as tf

print("🔧 Engine Fault Detection System")
print("=" * 40)

# Load models
try:
    scaler = joblib.load('saved_models/scaler.pkl')
    rf_model = joblib.load('saved_models/random_forest_model.pkl')
    cnn_model = tf.keras.models.load_model('saved_models/cnn_model.h5')
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

def extract_features(file_path, n_mfcc=13):
    """Extract audio features from WAV/MP3 file"""
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
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
        
        # Combine all features
        features = np.concatenate([
            mfccs_mean[:5],  # First 5 MFCCs
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
        print(f"❌ Error extracting features: {e}")
        return None

def predict(file_path, model_type='cnn'):
    """Make prediction on audio file"""
    print(f"\n🔍 Analyzing: {file_path}")
    print(f"🤖 Using model: {model_type}")
    
    features = extract_features(file_path)
    if features is None:
        return "Error", 0.0
    
    # Reshape and scale
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Make prediction
    if model_type.lower() == 'rf':
        prediction = rf_model.predict(features_scaled)
        probability = rf_model.predict_proba(features_scaled)
    else:  # CNN
        features_scaled_cnn = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
        prediction_proba = cnn_model.predict(features_scaled_cnn, verbose=0)
        prediction = (prediction_proba > 0.5).astype(int)
        probability = np.array([[1 - prediction_proba[0][0], prediction_proba[0][0]]])
    
    fault_prob = probability[0][1]
    result = "NORMAL" if prediction[0] == 0 else "FAULTY"
    
    return result, fault_prob

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else 'cnn'
        
        result, prob = predict(audio_file, model_type)
        
        print("\n" + "=" * 40)
        print("📊 RESULTS")
        print("=" * 40)
        print(f"Status: {result}")
        print(f"Fault Probability: {prob:.2%}")
        print(f"Confidence: {'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW'}")
        
        if result == "FAULTY":
            print("🚨 Action: Engine requires inspection!")
        else:
            print("✅ Action: Engine appears normal")
    else:
        print("\nUsage: python predict.py <audio_file> [model_type]")
        print("Example: python predict.py engine_sound.wav cnn")
        print("Example: python predict.py engine_sound.wav rf")
