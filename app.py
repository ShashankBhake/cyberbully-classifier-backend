import re
import pickle
import zipfile
import os
import shutil
import json
import nltk
from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from io import BytesIO

# Set NLTK data path to a writable directory
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

app = Flask(__name__)

# Load your saved vectorizer and model
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
loaded_model = pickle.load(open('cyber.pkl', 'rb'))

lemma = WordNetLemmatizer()

def preprocess(txt):
    # Remove timestamp pattern
    txt = re.sub(r'^\[\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}:\d{2} [APM]{2}\] ', '', txt)
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
    for line in text.split('\n'):
        line = line.replace('\u200e', '')  # Remove U+200E character
        if re.match(r'^\[\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}:\d{2} [APM]{2}\]', line):
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
    """Find the WhatsApp chat file in the zip archive"""
    # Look for files that match common WhatsApp chat export patterns
    chat_patterns = ['_chat.txt', 'WhatsApp Chat.txt', 'chat.txt']
    
    for filename in zip_ref.namelist():
        # Get just the file name without the path
        base_name = os.path.basename(filename)
        if any(pattern.lower() in base_name.lower() for pattern in chat_patterns):
            return filename
    return None

@app.route('/')
def home():
    return 'App is running.', 200

@app.route('/predict', methods=['GET'])
def predict():
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

@app.route('/upload', methods=['POST'])
def upload():
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

if __name__ == "__main__":
    app.run(debug=True)