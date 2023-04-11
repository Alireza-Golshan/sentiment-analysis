import random
import numpy as np
import csv
import tensorflow as tf
from sentiment_analysis import get_datasource
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt


def normalize_data(interview):
    text = interview.replace("answer", "").split('\n', 1)[1]
    return text[0:256]


vocab_size = 3000
embedding_dim = 16
max_length = 120
training_size = 22000
trunc_type = padding_type = 'post'
oov_tok = "<OOV>"

data_source = get_datasource()

test_data_store = []
train_data_store = []

with open('./train.csv', newline='', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    train_data_store = list(reader)

with open('./test.csv', newline='', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    test_data_store = list(reader)

training_sentences = []
training_labels = []
for item in train_data_store:
    training_sentences.append(item[1])
    training_labels.append(int(item[2]))

testing_sentences = []
testing_labels = []
for item in test_data_store:
    testing_sentences.append(item[1])
    testing_labels.append(int(item[2]))

print(len(training_sentences))
print(len(testing_sentences))
print(len(training_labels))
print(len(testing_labels))


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
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
num_epochs = 5

history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=1)
data = list(map(normalize_data, (list(data_source.values()))))
sentences = tokenizer.texts_to_sequences(data)
padded = pad_sequences(sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print("Shape of input data:", padded.shape)
lst = model.predict(padded)

p1 = float("{0:.2f}".format(float(lst[0])))
p2 = float("{0:.2f}".format(float(lst[1])))
p3 = float("{0:.2f}".format(float(lst[2])))
p4 = float("{0:.2f}".format(float(lst[3])))
p5 = float("{0:.2f}".format(float(lst[4])))
p6 = float("{0:.2f}".format(float(lst[5])))
p7 = float("{0:.2f}".format(float(lst[6])))
p8 = float("{0:.2f}".format(float(lst[7])))
p9 = float("{0:.2f}".format(float(lst[8])))

plt.style.use('ggplot')
x = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']
data = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, data, color='blue')
plt.xlabel("sentiment analysis")
plt.ylabel("Estimate")
plt.title("Results")
plt.xticks(x_pos, x)
plt.gca().set_ylim([0.0, 1.0])
plt.show()