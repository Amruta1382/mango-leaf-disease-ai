import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# -------------------------
# 1. Dataset Path
# -------------------------
dataset_path = "dataset"

# -------------------------
# 2. Image Preprocessing
# -------------------------
img_size = 224
batch_size = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# -------------------------
# 3. Load MobileNetV2 Base Model
# -------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size, img_size, 3)
)

base_model.trainable = False  # Freeze base model

# -------------------------
# 4. Add Custom Layers
# -------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# -------------------------
# 5. Compile Model
# -------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# 6. Train Model
# -------------------------
epochs = 5

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# -------------------------
# 7. Save Model
# -------------------------
if not os.path.exists("model"):
    os.makedirs("model")

model.save("model/mango_model.h5")

print("Model training complete and saved successfully!")