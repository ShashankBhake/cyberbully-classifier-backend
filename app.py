import re
import pickle
import zipfile
import os
import shutil
import json
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from io import BytesIO
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set NLTK data path to a writable directory
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

app = Flask(__name__)

# Enable CORS
CORS(app)

# Handle proxy headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Error handling
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

# Load models with error handling
try:
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    loaded_model = pickle.load(open('cyber.pkl', 'rb'))
    lemma = WordNetLemmatizer()
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def preprocess(txt):
    # Remove timestamp pattern
    txt = re.sub(r'\[?\d{2}/\d{2}/\d{2},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)\]?\s*-?\s*', '', txt)
    txt = re.sub('[^a-zA-Z]', ' ', txt).lower().split()
    txt = [lemma.lemmatize(word) for word in txt]
    return ' '.join(txt)

def is_word_in_category(word, category_index):
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = loaded_model.coef_[category_index]
    word_index = np.where(feature_names == word)[0]
    if len(word_index) == 0:
        return False
    return coefs[word_index][0] != 0

def predict_bullying_types(messages, type):
    labels = ['not_bullying', 'gender', 'religion', 'other_bullying', 'age', 'ethnicity']
    label_counts = {label: 0 for label in labels}
    matched_words = {label: set() for label in labels}
    
    for message in messages:
        if message.strip():  # Skip empty lines
            preprocessed = preprocess(message)
            vec = tfidf.transform([preprocessed])
            prediction = loaded_model.predict(vec)[0]
            label_counts[labels[prediction]] += 1
            for word in preprocessed.split():
                if is_word_in_category(word, prediction):
                    matched_words[labels[prediction]].add(word)
    
    matched_words = {label: list(words) for label, words in matched_words.items()}
    total_messages = len([msg for msg in messages if msg.strip()])  # Count non-empty messages
    final_score = {label: round((count / total_messages) * 100, 2) for label, count in label_counts.items()}
    final_score['total_messages'] = total_messages
    # final_score['matched_words'] = matched_words # Uncomment this line to include matched words in the response
    return final_score

def extract_messages(text):
    messages = []
    current_message = []

    timestamp_pattern = r'^\[?\d{2}/\d{2}/\d{2},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)\]?\s*-?\s*'
    
    for line in text.split('\n'):
        line = line.replace('\u200e', '')  # Remove U+200E character
        if re.match(timestamp_pattern, line):
            if current_message:
                messages.append(' '.join(current_message))
                current_message = []
        current_message.append(line)
    if current_message:
        messages.append(' '.join(current_message))
    return messages

def extract_instagram_messages(username, zip_data):
    messages = []
    try:
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            # Find message files for the username
            message_path = f'your_instagram_activity/messages/inbox/{username}'
            message_files = [f for f in zip_ref.namelist() 
                           if f.startswith(message_path) and f.endswith('.json')]
            
            for file_path in message_files:
                with zip_ref.open(file_path) as json_file:
                    data = json.load(json_file)
                    for message in data.get('messages', []):
                        if 'content' in message:
                            messages.append(message['content'])
    except Exception as e:
        print(f"Error processing Instagram messages: {str(e)}")
    return messages

def extract_facebook_messages(username, zip_data):
    messages = []
    try:
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            # Find message files for the username
            message_path = f'your_facebook_activity/messages/inbox/{username}'
            message_files = [f for f in zip_ref.namelist() 
                           if f.startswith(message_path) and f.endswith('.json')]
            
            for file_path in message_files:
                with zip_ref.open(file_path) as json_file:
                    data = json.load(json_file)
                    for message in data.get('messages', []):
                        if 'content' in message:
                            messages.append(message['content'])
    except Exception as e:
        print(f"Error processing Facebook messages: {str(e)}")
    return messages

def find_chat_file(zip_ref):
    """Find any text file in the zip archive"""
    for filename in zip_ref.namelist():
        # Get just the file name without the path
        base_name = os.path.basename(filename)
        if base_name.lower().endswith('.txt'):
            return filename
    return None

@app.route('/')
def home():
    return jsonify({'message': 'Cyberbullying Detection API', 'status': 'running'}), 200

@app.route('/predict', methods=['GET'])
def predict():
    try:
        user_text = request.args.get('message')
        if not user_text:
            return jsonify({'error': 'No message provided'}), 400
        
        preprocessed = preprocess(user_text)
        vec = tfidf.transform([preprocessed])
        prediction = loaded_model.predict(vec)[0]
        labels = ['not_bullying', 'gender', 'religion', 'other_bullying', 'age', 'ethnicity']
        result = labels[prediction]
        matched_words = [word for word in preprocessed.split() if result != 'not_bullying' and is_word_in_category(word, prediction)]
        
        return jsonify({'prediction': result, 'matched_words': matched_words}), 200
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        if 'type' not in request.form:
            return jsonify({'error': 'No type part'}), 400
        
        file = request.files['file']
        type = request.form['type']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if type.strip() == '':
            return jsonify({'error': 'No selected type'}), 400
        
        if type.strip() == 'whatsapp':
            if file and file.filename.endswith('.zip'):
                # Read zip file into memory
                zip_data = BytesIO(file.read())
                try:
                    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                        # Find chat file in zip
                        chat_file_path = find_chat_file(zip_ref)
                        if not chat_file_path:
                            return jsonify({'error': 'No WhatsApp chat file found in the zip'}), 400
                        
                        # Read the found chat file
                        chat_file = zip_ref.read(chat_file_path).decode('utf-8')
                        messages = extract_messages(chat_file)
                        if not messages:
                            return jsonify({'error': 'No messages found in chat file'}), 400
                        
                        final_score = predict_bullying_types(messages, 'whatsapp')
                        return jsonify(final_score), 200
                except Exception as e:
                    return jsonify({'error': f'Error processing zip file: {str(e)}'}), 400
            
            elif file and file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
                messages = extract_messages(text)
                final_score = predict_bullying_types(messages, 'whatsapp')
                return jsonify(final_score), 200
            
            else:
                return jsonify({'error': 'Invalid file format'}), 400
        
        elif type.strip() == 'instagram':
            if 'username' not in request.form:
                return jsonify({'error': 'No username part'}), 400
            
            username = request.form['username']
            
            if file and file.filename.endswith('.zip'):
                zip_data = BytesIO(file.read())
                messages = extract_instagram_messages(username, zip_data)
                if not messages:
                    return jsonify({'error': f'No messages found for the user {username}'}), 400
                
                final_score = predict_bullying_types(messages, 'instagram')
                return jsonify(final_score), 200
            else:
                return jsonify({'error': 'Invalid file format'}), 400
        
        elif type.strip() == 'facebook':
            if 'username' not in request.form:
                return jsonify({'error': 'No username part'}), 400
            
            username = request.form['username']
            
            if file and file.filename.endswith('.zip'):
                zip_data = BytesIO(file.read())
                messages = extract_facebook_messages(username, zip_data)
                if not messages:
                    return jsonify({'error': f'No messages found for the user {username}'}), 400
                
                final_score = predict_bullying_types(messages, 'facebook')
                return jsonify(final_score), 200
            else:
                return jsonify({'error': 'Invalid file format'}), 400
        
        else:
            return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    
    # In production, listen on all interfaces
    app.run(host='0.0.0.0', port=port)