from flask import Flask, render_template, request, send_from_directory
import os
import uuid
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['OUTPUT_FOLDER'] = 'static/outputs/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    torch_dtype=torch.float16
).to(device)
pipe.safety_checker = None 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(input_image, prompt):
    image = input_image.convert("RGB").resize((512, 512))
    
    return pipe(
        prompt=f"{prompt}, Ghibli style, anime artwork, vibrant colors, detailed background",
        image=image,
        strength=0.5, 
        guidance_scale=7.5,
        num_inference_steps=50
    ).images[0]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
            
        file = request.files['file']
        prompt = request.form.get('prompt', '')  
        
        if file.filename == '':
            return render_template('index.html', error="No selected file")
            
        if file and allowed_file(file.filename):
            input_img = Image.open(file.stream)
            output_img = process_image(input_img, prompt)
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
            
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{uuid.uuid4()}.png")
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"output_{uuid.uuid4()}.png")
            
            input_img.save(input_path)
            output_img.save(output_path)
            
            return render_template('index.html', 
                                original=input_path,
                                transformed=output_path)
        
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)