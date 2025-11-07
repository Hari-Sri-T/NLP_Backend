import tensorflow as tf

print("Loading the full .keras model...")
# 1. LOAD YOUR ORIGINAL MODEL
# --- Make sure this path is correct ---
model = tf.keras.models.load_model('model/best_multivariate_lstm.keras') 
print("Model loaded.")

# 2. CREATE THE CONVERTER
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. ADD SETTINGS (CRITICAL FOR LSTMS)
# This enables default optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This enables the special "ops" (like LSTM) to work
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite's built-in ops
    tf.lite.OpsSet.SELECT_TF_OPS    # TensorFlow's ops (for the LSTM)
]
converter._experimental_lower_tensor_list_ops = False
print("Converter settings applied.")

# 4. CONVERT THE MODEL
print("Converting model...")
tflite_model = converter.convert()
print("Conversion complete.")

# 5. SAVE THE NEW .TFLITE FILE
# --- This will save the new model to your model/ folder ---
with open('model/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Successfully saved model.tflite to /model folder.")