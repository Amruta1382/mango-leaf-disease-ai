from flask import Flask, render_template, request, redirect, flash
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# ==============================
# Load environment variables
# ==============================
load_dotenv()

# ==============================
# Initialize Flask app
# ==============================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# ==============================
# Load Model
# ==============================
model_path = os.path.join("model", "mango_model.h5")
model = load_model(model_path, compile=False)

# ==============================
# Class Names
# ==============================
class_names = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"]

# ==============================
# MongoDB Connection
# ==============================
client = MongoClient(os.getenv("MONGO_URI"))
db = client["mango_db"]
collection = db["predictions"]

# ==============================
# Home Route
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# Prediction Route
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file selected")
        return redirect("/")

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected")
        return redirect("/")

    filename = secure_filename(file.filename)
    filepath = os.path.join("static/uploads", filename)
    file.save(filepath)

    # Preprocess Image
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Save to MongoDB
    collection.insert_one({
        "filename": filename,
        "prediction": predicted_class,
        "date": datetime.now()
    })

    return render_template("result.html", prediction=predicted_class, image=filepath)

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)