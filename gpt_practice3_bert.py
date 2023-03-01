import tensorflow as tf
from transformers import BertTokenizer, TFAutoModel

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the BERT model
tf_model = TFAutoModel.from_pretrained('bert-base-uncased')

# Define the classification model
inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
bert_output = tf_model(inputs)[1]
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Prepare the training data
train_sentences = ['I love pizza', 'I hate broccoli', 'Pizza is the best', 'Broccoli is gross']
train_labels = [1, 0, 1, 0]
train_encodings = tokenizer(train_sentences, padding=True, truncation=True, max_length=128)

# Train the model
model.fit(train_encodings['input_ids'], train_labels, epochs=3, batch_size=2)

# Test the model
test_sentences = ['I like pizza', 'I don\'t like broccoli']
test_encodings = tokenizer(test_sentences, padding=True, truncation=True, max_length=128)
test_predictions = model.predict(test_encodings['input_ids'])
print(test_predictions)
