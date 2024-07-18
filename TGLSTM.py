import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess the text data
text = open('your_text_file.txt', 'r').read().lower()
chars = sorted(list(set(text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 100
step = 1
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

# Build the model
model = Sequential([
    LSTM(128, input_shape=(seq_length, len(chars))),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, batch_size=128, epochs=20)

# Generate text
start_index = np.random.randint(0, len(text) - seq_length - 1)
generated_text = text[start_index: start_index + seq_length]
print('Seed text:', generated_text)

for i in range(400):
    x = np.zeros((1, seq_length, len(chars)))
    for t, char in enumerate(generated_text):
        x[0, t, char_to_int[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = np.random.choice(len(chars), p=preds)
    next_char = int_to_char[next_index]

    generated_text += next_char
    generated_text = generated_text[1:]

print('Generated text:', generated_text)
