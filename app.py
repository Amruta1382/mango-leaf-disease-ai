from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import load_img, img_to_array
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flash messages

# Upload folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "mango_model.h5")
model = tf.keras.models.load_model(model_path)

# Class labels (same order as training)
class_names = ["Anthracnose", "Bacterial Canker", "Healthy", "Powdery Mildew"]

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["mango_disease_db"]
collection = db["predictions"]

# Helper function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Predict function
def predict_image(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        return predicted_class, round(confidence, 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and allowed_file(file.filename):
            # Use secure filename
            filename = secure_filename(file.filename)
            # Avoid overwriting by adding timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Predict
            prediction, confidence = predict_image(file_path)

            if prediction is not None:
                # Save prediction to MongoDB
                try:
                    collection.insert_one({
                        "filename": filename,
                        "prediction": prediction,
                        "confidence": confidence,
                        "timestamp": datetime.now()
                    })
                except Exception as e:
                    print(f"MongoDB insert error: {e}")
            else:
                flash("Prediction failed. Please try again.")

        else:
            flash("Invalid file type! Only png, jpg, jpeg allowed.")
            return redirect(request.url)

    return render_template("index.html", prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)