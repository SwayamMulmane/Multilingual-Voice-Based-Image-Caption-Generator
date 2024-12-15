from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/satyamdoijode/Desktop/Uploaded Images'
app.config['TRANSLATED_AUDIO'] = 'translated_audio.mp3'

# Initialize the vision model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_caption():
    try:
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return "No image file provided", 400

        file = request.files['image']

        if file.filename == '':
            return "No file selected", 400

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Ensure the upload folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file.save(file_path)

            # Predict caption
            caption = predict_caption(file_path)

            # Translate caption to the selected language
            language = request.form.get('language')
            if not language:
                return "Language not selected", 400

            translated_text = translate_text(caption, language)

            # Save translated text to audio file
            audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['TRANSLATED_AUDIO'])
            save_audio(translated_text, language, audio_file_path)

            # Pass image URL to template
            image_url = url_for('uploaded_image', filename=filename)

            return render_template('index.html', caption=caption, translated_text=translated_text, audio_url=url_for('serve_audio'), image_url=image_url)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

def predict_caption(image_path):
    try:
        image = Image.open(image_path)
    except Exception as e:
        return f"Error loading image: {str(e)}"

    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0]

def translate_text(text, target_language):
    try:
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        return f"Translation failed: {str(e)}"

def save_audio(text, language, audio_file_path):
    try:
        gtts_object = gTTS(text=text, lang=language, slow=False)
        gtts_object.save(audio_file_path)
    except Exception as e:
        raise ValueError(f"Error saving audio: {str(e)}")

@app.route('/audio')
def serve_audio():
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], app.config['TRANSLATED_AUDIO'], as_attachment=True)
    except Exception as e:
        return f"Error serving audio: {str(e)}", 500

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return f"Error serving image: {str(e)}", 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True, port=4545, use_reloader=False)
