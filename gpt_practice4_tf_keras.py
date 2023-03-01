import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the intent labels
labels = ["greeting", "goodbye", "thanks", "unknown"]

# Define the training data
train_data = [
    ("Hi there!", "greeting"),
    ("Hello!", "greeting"),
    ("Goodbye!", "goodbye"),
    ("See you later!", "goodbye"),
    ("Thanks!", "thanks"),
    ("Thank you!", "thanks")
]

# Tokenize the training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text for text, label in train_data])

# Convert the training data to sequences
train_sequences = tokenizer.texts_to_sequences([text for text, label in train_data])

# Pad the sequences to a fixed length
max_length = max(len(seq) for seq in train_sequences)
train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post')

# Convert the labels to one-hot encodings
label_encoder = {label: i for i, label in enumerate(labels)}
train_labels = [label_encoder[label] for text, label in train_data]
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(labels))

# Define the model architecture
model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length))
model.add(layers.Conv1D(64, 5, activation='relu', padding="same"))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_sequences, train_labels, epochs=50)

# Evaluate the model
test_data = [
    ("Hi there!", "greeting"),
    ("Goodbye!", "goodbye"),
    ("Thanks!", "thanks"),
    ("What's the weather like today?", "unknown")
]

test_sequences = tokenizer.texts_to_sequences([text for text, label in test_data])
test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')
test_labels = [label_encoder[label] for text, label in test_data]
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(labels))

loss, accuracy = model.evaluate(test_sequences, test_labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
