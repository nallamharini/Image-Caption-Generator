# app.py
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# Load pretrained model
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def generate_caption():
    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        try:
            image = Image.open(file.stream)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
    
    # Handle URL input
    elif 'url' in request.form:
        url = request.form['url']
        if not url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format'}), 400
        
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code != 200:
                return jsonify({'error': f'Failed to download image (HTTP {response.status_code})'}), 400
            
            image = Image.open(BytesIO(response.content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            return jsonify({'error': f'URL processing failed: {str(e)}'}), 400
    
    else:
        return jsonify({'error': 'No image provided'}), 400

    try:
        result = captioner(image)
        caption = result[0]['generated_text']
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)