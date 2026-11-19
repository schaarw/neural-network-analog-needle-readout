import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from google.colab import files, drive

# --- 1. OPTIONAL: GOOGLE DRIVE EINBINDEN ---
# Falls Ihre Bilder in Google Drive liegen, uncommenten Sie die folgenden Zeilen:
# drive.mount('/content/drive')
# IMAGE_DIR = "/content/drive/MyDrive/Colab Notebooks/gauge_images"

# Alternativ: Lokaler Ordner in der Colab-Session
IMAGE_DIR = "/content/gauge_images"

# Ordner erstellen, falls er nicht existiert (zum Hochladen via Dateimanager)
os.makedirs(IMAGE_DIR, exist_ok=True)
print(f"Bitte laden Sie Ihre Bilder in das Verzeichnis hoch: {IMAGE_DIR}")

# --- 2. CONFIGURATION ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 15

# --- 3. PARSE FILENAMES ---
# Erwartet Dateinamen, die mit dem Messwert beginnen, z. B. "42.5_img.jpg" oder "101.0.jpg"
records = []
if os.path.exists(IMAGE_DIR):
    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Regex sucht nach einer Zahl am Anfang des Dateinamens
            match = re.match(r"^([0-9]+\.?[0-9]*)", fname)
            if match:
                val = float(match.group(1))
                records.append({
                    "path": os.path.join(IMAGE_DIR, fname),
                    "value": val
                })

df = pd.DataFrame(records)
print(f"Gefundene Bilder mit gültigem Wert im Dateinamen: {len(df)}")
if len(df) > 0:
    display(df.head())
else:
    raise ValueError("Keine passenden Bilder gefunden! Bitte Dateinamen prüfen (z.B. '25.5_test.jpg').")

# --- 4. TF.DATA PIPELINE ---
def load_and_preprocess(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # MobileNetV2 Vorverarbeitung (-1 bis 1)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, tf.cast(label, tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((df['path'].values, df['value'].values))
dataset = dataset.shuffle(buffer_size=len(df))
dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Train/Validation Split (80% / 20%)
val_size = max(1, int(len(df) * 0.2))
val_ds = dataset.take(val_size // BATCH_SIZE + 1)
train_ds = dataset.skip(val_size // BATCH_SIZE + 1)

# --- 5. MODELL ERSTELLEN (Transfer Learning für Regression) ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Basismodell einfrieren

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='linear')  # Linearer Output für kontinuierliche Werte (Regression)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae'] # Mean Absolute Error zur besseren Lesbarkeit im Log
)

# --- 6. TRAINING ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- 7. TFLITE KONVERTIERUNG & DOWNLOAD ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optional: Quantisierung für kleinere Dateigröße und bessere Performance auf Edge-Geräten
converter.optimizations = [tf.lite.Optimize.DEFAULT_SIZE]

tflite_model = converter.convert()

model_filename = 'analog_gauge_model.tflite'
with open(model_filename, 'wb') as f:
    f.write(tflite_model)

print(f"Modell erfolgreich exportiert: {model_filename}")
files.download(model_filename)