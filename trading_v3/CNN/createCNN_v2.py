from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, concatenate
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import pandas as pd

# Define the model
def create_model():
    # Input for CS image
    cs_input = Input(shape=(224, 224, 3), name='CS_Input')  # Resized input
    cs_model = Conv2D(32, (3, 3), activation='relu')(cs_input)
    cs_model = MaxPooling2D((2, 2))(cs_model)
    cs_model = Conv2D(64, (3, 3), activation='relu')(cs_model)
    cs_model = MaxPooling2D((2, 2))(cs_model)
    cs_model = Flatten()(cs_model)

    # Input for RSI image
    rsi_input = Input(shape=(224, 224, 3), name='RSI_Input')  # Resized input
    rsi_model = Conv2D(32, (3, 3), activation='relu')(rsi_input)
    rsi_model = MaxPooling2D((2, 2))(rsi_model)
    rsi_model = Conv2D(64, (3, 3), activation='relu')(rsi_model)
    rsi_model = MaxPooling2D((2, 2))(rsi_model)
    rsi_model = Flatten()(rsi_model)

    # Combine features from both inputs
    combined = concatenate([cs_model, rsi_model])

    # Fully connected layers
    fc = Dense(128, activation='relu')(combined)
    fc = Dropout(0.5)(fc)
    fc = Dense(64, activation='relu')(fc)
    fc = Dropout(0.5)(fc)

    # Output layer
    output = Dense(1, activation='tanh', name='Output')(fc)  # Outputs value between -1 and 1

    # Create the model
    model = Model(inputs=[cs_input, rsi_input], outputs=output)
    return model

# Utility function for preprocessing images
def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path)  # Load the image
    image = image.resize(target_size)  # Resize dynamically
    image_array = img_to_array(image)  # Convert to array
    image_array = image_array / 255.0  # Normalize to range [0, 1]
    return image_array

# Load dataset from CSV
def load_dataset(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Extract image paths and labels
    cs_image_paths = df['CS'].values
    rsi_image_paths = df['RSI'].values
    choices = df['Choice'].values

    # Preprocess the images
    cs_images = [preprocess_image(path) for path in cs_image_paths]
    rsi_images = [preprocess_image(path) for path in rsi_image_paths]

    return tf.convert_to_tensor(cs_images), tf.convert_to_tensor(rsi_images), tf.convert_to_tensor(choices)

# Path to your CSV file
csv_path = '/workspaces/python/tradeBotDataset.csv'

# Load the dataset
cs_images, rsi_images, choices = load_dataset(csv_path)

import numpy as np
from sklearn.model_selection import train_test_split

# Convert TensorFlow tensors to NumPy arrays
cs_images_np = cs_images.numpy()
rsi_images_np = rsi_images.numpy()
choices_np = choices.numpy()

# Split the dataset into training and testing sets
cs_images_train, cs_images_test, rsi_images_train, rsi_images_test, choices_train, choices_test = train_test_split(
    cs_images_np, rsi_images_np, choices_np, test_size=0.2, random_state=42
)

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    [cs_images_train, rsi_images_train],  # Training inputs
    choices_train,  # Training targets
    batch_size=32,
    epochs=50,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate([cs_images_test, rsi_images_test], choices_test)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Save the model after training
model.save('/workspaces/python/CNN/stock_prediction_model_v2.keras')
print("Model saved successfully!")
