import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Masking
from tensorflow.keras.layers import TimeDistributed, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import transformers
from transformers import TFAutoModel, AutoTokenizer

import sys
print("getrecursionlimit", sys.getrecursionlimit())
sys.setrecursionlimit(20000)
print("update recursion")

# Load the pre-trained Transformer model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = TFAutoModel.from_pretrained(model_name)
print("load")

# Define the input shape
max_length = 128

# Define the model architecture
input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
embedding_layer = bert(input_ids, attention_mask=attention_mask)[0]
x = LSTM(64)(embedding_layer)
x = Dropout(0.2)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=[input_ids, attention_mask], outputs=outputs)

# Compile the model
learning_rate = 0.001  # 2e-5
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
num_epochs = 10
batch_size = 32

# Prepare the training data
X_train = ["This is a positive sentence.", "This is a negative sentence."]
y_train = [0, 1]

# Tokenize the training data
encoded_input = tokenizer(X_train, padding=True, truncation=True, max_length=max_length, return_tensors="tf")

# Convert labels to one-hot vectors
y_train = to_categorical(y_train, num_classes=2)

# Train the model
model.fit([encoded_input["input_ids"], encoded_input["attention_mask"]], y_train, epochs=num_epochs, batch_size=batch_size)
