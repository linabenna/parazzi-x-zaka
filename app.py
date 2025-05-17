from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the model
model = load_model('model/tile_classifier_resnet18_more_data.pth')
class_names = ["ADENIA MIEL, ANTIBES GREY, ARTICWOOD ARGENT, AZUL CIELO, BLEND CONCRETE IRON, BRUNSWICH ACERO, CalacattaBright, IRAZU ANTRACITA, LILAC GRIGIO SIENA_LUX, MONTBLANC ANTRACITA, NEXSIDEBLUEPULIDO, NIEVE VERDE, PATMOS PULIDO, STONE GREY, VANGLIHPULIDO"] 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))  # adjust size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        return render_template('index.html', label=predicted_class, image_path=filepath)

    return render_template('index.html', label=None)

if __name__ == '__main__':
    app.run(debug=True)
