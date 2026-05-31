import os
import subprocess
import uuid
import numpy as np
import tensorflow as tf
import librosa
import noisereduce as nr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'apikeyhere'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///put save path here'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 3
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_LENGTH
FRAME_LENGTH = 255
FRAME_STEP = 128

tflite_model_path = "put save path here"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Input shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

class Spectrogram:
    def get_spectrogram(self, waveform):
        waveform = waveform[:SAMPLES_PER_SEGMENT] if len(waveform) >= SAMPLES_PER_SEGMENT else tf.concat([waveform, tf.zeros([SAMPLES_PER_SEGMENT - len(waveform)])], 0)
        spectrogram = tf.signal.stft(waveform, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)
        spectrogram = tf.abs(spectrogram)[..., tf.newaxis]
        spectrogram = tf.image.resize(spectrogram, [379, 128])
        return spectrogram

    def clean_waveform(self, waveform, sr):
        try:
            waveform_np = waveform.numpy()
            cleaned = nr.reduce_noise(y=waveform_np, sr=float(sr), stationary=False)
            return tf.convert_to_tensor(cleaned, dtype=tf.float32)
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return waveform

def preprocess_and_predict(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        print(f"Original Audio Stats: Mean={np.mean(audio)}, Std={np.std(audio)}, Max={np.max(audio)}, Min={np.min(audio)}, Sample Rate={sr}, Length={len(audio)}")

        # Check for silent audio
        if np.max(np.abs(audio)) < 1e-4:
            print("Warning: Audio amplitude too low, likely silent or corrupted")
            return "Error: Audio too quiet or silent", None

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = np.pad(audio, (0, max(0, SAMPLES_PER_SEGMENT - len(audio))))[:SAMPLES_PER_SEGMENT]

        print(f"Processed Audio Stats: Mean={np.mean(audio)}, Std={np.std(audio)}, Max={np.max(audio)}, Min={np.min(audio)}, Length={len(audio)}")

        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        # Temporarily bypass noise reduction
        # audio_tensor = Spectrogram().clean_waveform(audio_tensor, SAMPLE_RATE)
        spectrogram = Spectrogram().get_spectrogram(audio_tensor)
        print(f"Spectrogram shape: {spectrogram.shape}, Mean={tf.reduce_mean(spectrogram).numpy()}, Std={tf.math.reduce_std(spectrogram).numpy()}")

        spectrogram = tf.expand_dims(spectrogram, axis=0)
        spectrogram = tf.cast(spectrogram, dtype=input_details[0]['dtype'])
        # Set input tensor only once
        interpreter.set_tensor(input_details[0]['index'], spectrogram.numpy())
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        print(f"Raw prediction: {prediction}")

        probabilities = tf.nn.softmax(prediction[0]).numpy()
        predicted_class_index = tf.argmax(probabilities).numpy()
        print(f"Probabilities: {probabilities}, Predicted index: {predicted_class_index}")

        class_mapping = {0: 'Good', 1: 'Broken', 2: 'Heavy_loaded'}
        return class_mapping.get(predicted_class_index, 'Unknown'), probabilities

    except Exception as e:
        import traceback
        print("Error during prediction:\n", traceback.format_exc())
        return f"Error: {str(e)}", None

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    original_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join('Uploads', original_filename)
    file.save(file_path)

    if file.filename.endswith('.webm'):
        wav_path = file_path.replace('.webm', '.wav')
        ffmpeg_path = r"put save path here"
        try:
            # Convert to 16000 Hz, mono, 16-bit PCM
            result = subprocess.run([ffmpeg_path, "-y", "-i", file_path, "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), wav_path],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stderr.decode())
            if not os.path.exists(wav_path):
                raise FileNotFoundError("FFmpeg failed to create WAV.")
            # Verify WAV file properties
            audio_check, sr_check = librosa.load(wav_path, sr=None, mono=True)
            print(f"Converted WAV Stats: Sample Rate={sr_check}, Channels={1 if audio_check.ndim == 1 else audio_check.shape[1]}, Duration={len(audio_check)/sr_check}s, Max={np.max(audio_check)}")
            file_path = wav_path
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    predicted_class, probabilities = preprocess_and_predict(file_path)
    if 'Error' in predicted_class:
        return jsonify({'error': predicted_class}), 500

    # Clean up files
    if os.path.exists(file_path):
        os.remove(file_path)
    if file.filename.endswith('.webm') and os.path.exists(file_path.replace('.wav', '.webm')):
        os.remove(file_path.replace('.wav', '.webm'))

    return jsonify({
        'predicted_class': predicted_class,
        'probabilities': probabilities.tolist() if probabilities is not None else []
    })

if __name__ == '__main__':
    os.makedirs('Uploads', exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False)