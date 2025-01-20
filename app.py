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

# Set NLTK data path to a writable directory
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

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

def extract_instagram_messages(username):
    messages = []
    base_path = os.path.join('extracted_files', 'your_instagram_activity', 'messages', 'inbox')
    user_folders = []
    for folder_name in os.listdir(base_path):
        if folder_name.startswith(username):
            user_folders.append(folder_name)

    if len(user_folders) == 0:
        return messages
    for user_folder in user_folders:
        user_path = os.path.join(base_path, user_folder)
        for file_name in os.listdir(user_path):
            if file_name.startswith('message_') and file_name.endswith('.json'):
                with open(os.path.join(user_path, file_name), 'r') as json_file:
                    data = json.load(json_file)
                    for message in data.get('messages', []):
                        if 'content' in message:
                            messages.append(message['content'])
    return messages

def extract_facebook_messages(username):
    messages = []
    base_path = os.path.join('extracted_files', 'your_facebook_activity', 'messages', 'inbox')
    user_folders = []
    for folder_name in os.listdir(base_path):
        if folder_name.startswith(username):
            user_folders.append(folder_name)
    if len(user_folders) == 0:
        return messages
    for user_folder in user_folders:
        user_path = os.path.join(base_path, user_folder)
        for file_name in os.listdir(user_path):
            if file_name.startswith('message_') and file_name.endswith('.json'):
                with open(os.path.join(user_path, file_name), 'r') as json_file:
                    data = json.load(json_file)
                    for message in data.get('messages', []):
                        if 'content' in message:
                            messages.append(message['content'])
                            
    return messages

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
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall('extracted_files')
            
            chat_file_path = os.path.join('extracted_files', '_chat.txt')
            if not os.path.exists(chat_file_path):
                shutil.rmtree('extracted_files')
                return jsonify({'error': 'No _chat.txt file found in the zip'}), 400
            
            with open(chat_file_path, 'r') as chat_file:
                text = chat_file.read()
            
            messages = extract_messages(text)
            if not messages:  # Check if messages were extracted
                shutil.rmtree('extracted_files')
                return jsonify({'error': 'No messages found in _chat.txt'}), 400
            
            final_score = predict_bullying_types(messages, 'whatsapp')
            
            # Clean up extracted files
            shutil.rmtree('extracted_files')
            
            return jsonify(final_score), 200
        
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
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall('extracted_files')
            
            messages = extract_instagram_messages(username)
            if not messages:  # Check if messages were extracted
                shutil.rmtree('extracted_files')
                return jsonify({'error': f'No messages found for the user {username}'}), 400
            
            final_score = predict_bullying_types(messages, 'instagram')
            
            # Clean up extracted files
            shutil.rmtree('extracted_files')
            
            return jsonify(final_score), 200

        else:
            return jsonify({'error': 'Invalid file format'}), 400
        
    elif type.strip() == 'facebook':
        if 'username' not in request.form:
            return jsonify({'error': 'No username part'}), 400
        
        username = request.form['username']
        
        if file and file.filename.endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall('extracted_files')
            
            messages = extract_facebook_messages(username)
            if not messages:
                shutil.rmtree('extracted_files')
                return jsonify({'error': f'No messages found for the user {username}'}), 400
            
            final_score = predict_bullying_types(messages, 'facebook')
            
            # Clean up extracted files
            shutil.rmtree('extracted_files')
            
            return jsonify(final_score), 200
        
        else:
            return jsonify({'error': 'Invalid file format'}), 400
    
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == "__main__":
    app.run(debug=True)