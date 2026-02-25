import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# ===============================
# 1. CONFIGURATION
# ===============================
DATASET_PATH = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "mango_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001

# ===============================
# 2. CHECK DATASET
# ===============================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset folder '{DATASET_PATH}' not found!")

print("Dataset found successfully.")

# ===============================
# 3. IMAGE PREPROCESSING
# ===============================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("Training and validation data loaded.")

# Save class names (VERY IMPORTANT for Flask app)
class_names = list(train_data.class_indices.keys())
print("Class Names:", class_names)

# ===============================
# 4. LOAD PRETRAINED MODEL
# ===============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Freeze base model

# ===============================
# 5. BUILD CUSTOM MODEL
# ===============================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

# ===============================
# 6. COMPILE MODEL
# ===============================
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# ===============================
# 7. TRAIN MODEL
# ===============================
print("\nStarting training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

print("\nTraining complete.")

# ===============================
# 8. SAVE MODEL & CLASS NAMES
# ===============================
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model.save(MODEL_PATH)

# Save class names to JSON
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

print("\nModel saved at:", MODEL_PATH)
print("Class names saved at:", CLASS_NAMES_PATH)
print("\nEverything completed successfully! ")