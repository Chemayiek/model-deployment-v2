import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
model_path = os.path.join(script_dir, 'models/keras_model.h5')
model = load_model(model_path)

# Class indices used during training
class_indices = {'Tomato___Bacterial_spot': 0, 'Tomato___Early_blight': 1, 'Tomato___Late_blight': 2,
                 'Tomato___Leaf_Mold': 3, 'Tomato___Septoria_leaf_spot': 4,
                 'Tomato___Spider_mites Two-spotted_spider_mite': 5, 'Tomato___Target_Spot': 6,
                 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 7, 'Tomato___Tomato_mosaic_virus': 8,
                 'Tomato___healthy': 9, 'tesst': 10}

print("Class Indices:", class_indices)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image for the model
            img = Image.open(filepath)
            img = img.resize((128, 128))  # Match the size used during training
            img_array = np.array(img) / 255.0  # Convert image to numpy array and normalize
            img_array = img_array.reshape((1, 128, 128, 3))  # Reshape for model input

            # Make prediction using the model
            prediction = model.predict(img_array)
            class_index = prediction.argmax()
            
            # Map predicted class index to class name
            predicted_class = [k for k, v in class_indices.items() if v == class_index][0]

            # Delete the uploaded image
            os.remove(filepath)

            return render_template('result.html', predicted_class=predicted_class)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
