# Import necessary libraries
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Add, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam

# Define dataset paths
IMAGES_PATH = '/kaggle/input/flickr8k/Images'
TEXT_FILE = '/kaggle/input/flickr8k/captions.txt'

# Function to load captions and process data
def load_captions_data(filename):
    with open(filename, "r", encoding="utf-8") as file:
        caption_data = file.readlines()[1:]  # Skip header

    caption_mapping = {}
    text_data = []
    
    for line in caption_data:
        parts = line.strip().split(",", 1)
        if len(parts) < 2:
            continue
        img_name, caption = parts
        img_name = os.path.join(IMAGES_PATH, img_name.strip())

        tokens = caption.strip().split()
        if not (5 <= len(tokens) <= 25):
            continue
        
        formatted_caption = f"<start> {caption.strip()} <end>"
        text_data.append(formatted_caption)

        if img_name in caption_mapping:
            caption_mapping[img_name].append(formatted_caption)
        else:
            caption_mapping[img_name] = [formatted_caption]

    return caption_mapping, text_data

# Load dataset
captions_mapping, text_data = load_captions_data(TEXT_FILE)

# Display basic dataset info
def display_dataset_info(captions_mapping):
    print(f"Total images in dataset: {len(captions_mapping)}\n")
    sample_captions = random.sample(list(captions_mapping.items()), min(5, len(captions_mapping)))
    print("Sample Captions:")
    for i, (img_name, captions) in enumerate(sample_captions):
        print(f"{i+1}. {img_name}: {captions[0]}")

display_dataset_info(captions_mapping)

# Tokenization setup
MAX_LENGTH = 40
VOCABULARY_SIZE = 15000

tokenizer = Tokenizer(num_words=VOCABULARY_SIZE, filters='', oov_token='<unk>')
tokenizer.fit_on_texts(text_data)

# Convert captions into sequences
sequences = tokenizer.texts_to_sequences(text_data)
sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')

print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Sample caption sequence: {sequences[0]}")

# Split dataset into training and validation
img_keys = list(captions_mapping.keys())
random.shuffle(img_keys)

split_index = int(len(img_keys) * 0.8)
train_img_keys = img_keys[:split_index]
val_img_keys = img_keys[split_index:]

def prepare_image_caption_pairs(img_keys, captions_mapping):
    img_paths, captions = [], []
    for img_key in img_keys:
        img_paths.extend([img_key] * len(captions_mapping[img_key]))
        captions.extend(captions_mapping[img_key])
    return img_paths, captions

train_imgs, train_captions = prepare_image_caption_pairs(train_img_keys, captions_mapping)
val_imgs, val_captions = prepare_image_caption_pairs(val_img_keys, captions_mapping)

print(f"Training dataset size: {len(train_imgs)}")
print(f"Validation dataset size: {len(val_imgs)}")

# Convert captions to sequences
def preprocess_captions(captions):
    sequences = tokenizer.texts_to_sequences(captions)
    return pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')

train_captions = preprocess_captions(train_captions)
val_captions = preprocess_captions(val_captions)

# Load InceptionV3 model
inception_model = InceptionV3(include_top=False, weights='imagenet')
inception_model.trainable = False

feature_extractor = Model(inputs=inception_model.input, outputs=GlobalAveragePooling2D()(inception_model.output))

def extract_features(img_paths, feature_extractor):
    features = []
    for img_path in img_paths:
        img = load_img(img_path, target_size=(299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        feature = feature_extractor.predict(img)
        features.append(feature)
    return np.vstack(features)

train_features = extract_features(train_imgs, feature_extractor)
val_features = extract_features(val_imgs, feature_extractor)

np.save('train_features.npy', train_features)
np.save('val_features.npy', val_features)
print("Features saved successfully.")

# Define image captioning model
def build_captioning_model(vocab_size, max_length, feature_dim):
    img_input = Input(shape=(feature_dim,))
    img_features = Dense(256, activation='relu')(img_input)

    cap_input = Input(shape=(max_length,))
    cap_embedding = Embedding(vocab_size, 256)(cap_input)
    cap_lstm = LSTM(256)(cap_embedding)

    merged = Add()([img_features, cap_lstm])
    output = Dense(vocab_size, activation='softmax')(merged)

    return Model(inputs=[img_input, cap_input], outputs=output)

captioning_model = build_captioning_model(VOCABULARY_SIZE, MAX_LENGTH, 2048)
captioning_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
captioning_model.summary()

# Train the model
captioning_model.fit([train_features, train_captions], train_captions, epochs=10, batch_size=32)

# Predict caption
def predict_caption(image_path, model, tokenizer):
    img_feature = extract_features([image_path], feature_extractor)
    caption_sequence = tokenizer.texts_to_sequences(["<start>"])
    for _ in range(MAX_LENGTH):
        prediction = model.predict([img_feature, np.array(caption_sequence)])
        predicted_word_idx = np.argmax(prediction[0])
        predicted_word = tokenizer.index_word.get(predicted_word_idx, '')
        if predicted_word in ['<end>', '']:
            break
        caption_sequence.append([predicted_word_idx])
    return ' '.join([tokenizer.index_word.get(i, '') for i in caption_sequence[0]])

# Example usage
image_path = '/path/to/new/image.jpg'  # Replace with actual image path
predicted_caption = predict_caption(image_path, captioning_model, tokenizer)
print(f"Predicted Caption: {predicted_caption}")
