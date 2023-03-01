from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

# Define the tokenizer and load the pre-trained model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Define the training data
train_data = [("What time does the train leave?", "schedule"),
              ("How much does the product cost?", "pricing"),
              ("Can you help me with my account?", "support")]

# Tokenize the training data and create input tensors
train_texts = [text for text, label in train_data]
train_labels = [label for text, label in train_data]
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).batch(16)

# Train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss)
model.fit(train_dataset, epochs=3)

# Test the model
test_data = ["When does the store close?", "What is your return policy?", "How do I contact customer service?"]
test_encodings = tokenizer(test_data, truncation=True, padding=True)
test_dataset = tf.data.Dataset.from_tensor_slices(dict(test_encodings)).batch(16)
outputs = model.predict(test_dataset)
predicted_labels = [tokenizer.decode(logits.argmax(axis=-1)) for logits in outputs[0]]
print(predicted_labels)
