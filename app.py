from flask import Flask, render_template, request, redirect, flash
import keras
import numpy as np
import os
from keras.utils import load_img, img_to_array
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
app.secret_key = "supersecretkey"

# ==============================
# Upload folder setup
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ==============================
# Load trained model
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "mango_model.keras")

# Important: safe_mode=False helps avoid config conflicts
model = keras.models.load_model(model_path, compile=False)

# Class labels
class_names = ["Anthracnose", "Bacterial Canker", "Healthy", "Powdery Mildew"]

# ==============================
# MongoDB Connection
# ==============================
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["mango_disease_db"]
collection = db["predictions"]

# ==============================
# Helper Functions
# ==============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
        print("Prediction error:", e)
        return None, None

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{filename}"

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            prediction, confidence = predict_image(file_path)

            if prediction:
                try:
                    collection.insert_one({
                        "filename": filename,
                        "prediction": prediction,
                        "confidence": confidence,
                        "timestamp": datetime.now()
                    })
                except Exception as e:
                    print("MongoDB error:", e)
            else:
                flash("Prediction failed. Try again.")

        else:
            flash("Upload PNG, JPG or JPEG only.")
            return redirect(request.url)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)

# ==============================
# Run
# ==============================

    if __name__ == "__main__":
     app.run()