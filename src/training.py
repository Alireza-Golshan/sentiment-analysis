import numpy as np
import csv
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

vocab_size = 27000
embedding_dim = 16
max_length = 120
training_size = 22000
trunc_type = padding_type = 'post'
oov_tok = "<OOV>"

with open('../mental-health-from-social-media-sites.csv', newline='') as f:
    reader = csv.reader(f)
    datastore = list(reader)

sentences = []
labels = []
for item in datastore:
    sentences.append(item[0])
    labels.append(int(item[1]))

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
num_epochs = 30
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=1)

lst = ["Nice drawing", "very awful"]

sentences = tokenizer.texts_to_sequences(lst)
padded = pad_sequences(sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(model.predict(padded))
