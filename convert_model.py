"""Convert .keras model to TF.js graph model format
Fixes numpy compatibility issue with tensorflowjs
"""
import numpy as np

# Monkey-patch numpy for tensorflowjs compatibility
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'str'):
    np.str = str

import tensorflow as tf
import tensorflowjs as tfjs
import os, shutil

MODEL_PATH = 'best_cloud_densenet.keras'
OUTPUT_DIR = 'model'

print(f'Loading {MODEL_PATH}...')
model = tf.keras.models.load_model(MODEL_PATH)
print(f'✅ Model loaded: {model.name}')
print(f'   Input shape: {model.input_shape}')
print(f'   Output shape: {model.output_shape}')

# Clear output dir
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# Convert directly using tensorflowjs
print(f'\nConverting to TF.js graph model...')
tfjs.converters.save_keras_model(model, OUTPUT_DIR)
print('✅ Conversion complete!')

print(f'\nFiles in {OUTPUT_DIR}:')
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f'  {f} ({size / 1024:.1f} KB)')

# Check format
import json
with open(os.path.join(OUTPUT_DIR, 'model.json')) as fp:
    meta = json.load(fp)
    fmt = meta.get('format', 'unknown')
    print(f'\nModel format: {fmt}')
