import time
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import nltk
from nltk.corpus import words
import os
import pyttsx3
import io
from flask import send_file

nltk.download('words')

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Initialize MediaPipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Labels (A-Z + SPACE + SUBMIT)
class_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ["SPACE", "SUBMIT"]

# Define ROI
ROI_X, ROI_Y, ROI_W, ROI_H = 400, 0, 639, 220

# Store states
detected_letters = []
created_sentence = []  # Store words being created
submitted_words = []
submitted_sentence = ""
latest_keypoints = np.zeros(63)

# Load words corpus
word_list = set(words.words())

def extract_keypoints_from_image(image, landmarks):
    """Extract hand keypoints from detected landmarks only if inside ROI."""
    keypoints = []
    for lm in landmarks.landmark:
        abs_x, abs_y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
        if ROI_X <= abs_x <= ROI_X + ROI_W and ROI_Y <= abs_y <= ROI_Y + ROI_H:
            keypoints.append([lm.x, lm.y, lm.z])
        else:
            return np.zeros(63)
    return np.array(keypoints).flatten()

def generate_frames():
    """Generator function for webcam frames."""
    global latest_keypoints
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        latest_keypoints = np.zeros(63)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                latest_keypoints = extract_keypoints_from_image(frame, hand_landmarks)
                if latest_keypoints.sum() != 0:
                    break

        cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (255, 0, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/favicon.ico')
def favicon():
    return '', 200  # Return an empty response with status 204 (No Content)

@app.route('/submit_sentence', methods=['POST'])
def submit_sentence():
    """Handle manual text submission."""
    global created_sentence, submitted_words, submitted_sentence
    
    data = request.get_json()
    manual_text = data.get("manual_text", "").strip()
    
    # Process existing created_sentence
    if created_sentence:
        sentence = " ".join(created_sentence)
        submitted_words.append(sentence)
        submitted_sentence += sentence + " "
        created_sentence = []  # Reset created sentence
    
    # Add manual text if provided
    if manual_text:
        submitted_words.append(manual_text)
        submitted_sentence += manual_text + " "
    
    return jsonify({
        "message": "Sentence submitted successfully!",
        "created_sentence": "",
        "submitted_words": submitted_words,
        "submitted_sentence": submitted_sentence.strip()
    })



# Initialize TTS engine
from gtts import gTTS
from googletrans import Translator

# Initialize translator
translator = Translator()

# Supported languages for TTS and Translation
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi"
}

@app.route('/speak', methods=['POST'])
def speak():
    """Translate text and convert it to speech in the selected language."""
    data = request.get_json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")  # Default to English if not specified

    if not text:
        return jsonify({"error": "No text provided!"}), 400
    
    if lang not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Unsupported language: {lang}. Choose from {list(SUPPORTED_LANGUAGES.keys())}"}), 400

    # Translate text if the input is in English and the target language is different
    if lang != "en":
        translated_text = translator.translate(text, src="en", dest=lang).text
    else:
        translated_text = text

    # Convert translated text to speech
    tts = gTTS(text=translated_text, lang=lang)

    # Save to a temporary file
    temp_filename = "temp_audio.mp3"
    tts.save(temp_filename)

    # Read the file into memory
    audio_stream = io.BytesIO()
    with open(temp_filename, "rb") as f:
        audio_stream.write(f.read())

    audio_stream.seek(0)  # Reset pointer to start of stream

    return send_file(audio_stream, mimetype="audio/mp3", as_attachment=False)





@app.route('/register_letter', methods=['POST'])
def register_letter():
    """Register a detected letter when Space key is pressed."""
    global detected_letters, created_sentence, submitted_words, submitted_sentence, latest_keypoints

    if latest_keypoints.sum() != 0:
        input_data = latest_keypoints.reshape(1, 63, 1)
        prediction = model.predict(input_data, verbose=0)
        predicted_label = class_labels[np.argmax(prediction)]

        if predicted_label == "SPACE":
            if detected_letters:
                word = "".join(detected_letters)
                created_sentence.append(word)  # Add the word to created sentence
                detected_letters = []  # Reset detected letters

                #speak_text(submitted_sentence.strip())

            return jsonify({
                "letter": "SPACE",
                "letters": "",
                "created_sentence": " ".join(created_sentence),
                "submitted_words": submitted_words,
                "submitted_sentence": submitted_sentence.strip()
            })

        elif predicted_label == "SUBMIT":
            if created_sentence:
                sentence = " ".join(created_sentence)
                submitted_words.append(sentence)
                submitted_sentence += sentence + " "
                created_sentence = []  # Reset created sentence
            return jsonify({
                "letter": "SUBMIT",
                "letters": "",
                "created_sentence": "",
                "submitted_words": submitted_words,
                "submitted_sentence": submitted_sentence.strip(),
                "action": "SUBMITTED"
            })

        else:
            detected_letters.append(predicted_label)
            return jsonify({
                "letter": predicted_label,
                "letters": "".join(detected_letters),
                "created_sentence": " ".join(created_sentence),
                "submitted_words": submitted_words,
                "submitted_sentence": submitted_sentence.strip()
            })

    return jsonify({
        "letter": "No hand detected",
        "letters": "".join(detected_letters),
        "created_sentence": " ".join(created_sentence),
        "submitted_words": submitted_words,
        "submitted_sentence": submitted_sentence.strip()
    })




@app.route('/suggest')
def suggest():
    """Return detected letters and word suggestions."""
    global detected_letters, created_sentence, submitted_words
    current_sequence = "".join(detected_letters).lower()

    # Debugging Statements
    print(f"Current sequence: {current_sequence}")
    print(f"Total words loaded: {len(word_list)}")

    if not current_sequence:
        return jsonify({
            "letters": "",
            "suggestions": [],
            "created_sentence": " ".join(created_sentence),
            "submitted_words": submitted_words
        })

    # Ensure word_list contains valid words
    suggested_words = [word for word in word_list if word.startswith(current_sequence)]
    print(f"Sample words: {list(word_list)[:10]}")
    # Debugging: Print found words
    print(f"Suggested words (before sorting): {suggested_words[:10]}")
    


    # Sort by length and limit to 5 suggestions
    suggested_words = sorted(suggested_words, key=len)[:10]

    return jsonify({
        "letters": current_sequence.upper(),
        "suggestions": suggested_words,
        "created_sentence": " ".join(created_sentence),
        "submitted_words": submitted_words
    })


@app.route('/clear_detections', methods=['POST'])
def clear_detections():
    """Clear detected letters."""
    global detected_letters
    detected_letters = []
    return jsonify({
        "message": "Detections cleared successfully!",
        "letters": "",
        "created_sentence": " ".join(created_sentence),
        "submitted_words": submitted_words
    })

@app.route('/clear_all', methods=['POST'])
def clear_all():
    global detected_letters, created_sentence, submitted_words, submitted_sentence
    detected_letters = []  # Reset letters
    created_sentence = []  # Reset sentence
    submitted_words = []   # Reset submitted words
    submitted_sentence = ""  # Reset submitted sentence
    return jsonify({
        "message": "All data cleared successfully!",
        "letters": "",
        "created_sentence": "",
        "submitted_words": [],
        "submitted_sentence": ""
    })



if __name__ == '__main__':
    app.run(debug=True)