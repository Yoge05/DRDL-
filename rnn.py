# RNN_Text_Generation_Friend_Unique.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------- TRAINING TEXT (CHANGED) ----------------
text = "Artificial intelligence enables machines to learn patterns from data"

# ---------------- CHARACTER PROCESSING ----------------
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# ---------------- SEQUENCE SETTINGS ----------------
SEQ_LEN = 6          # 🔁 increased
GEN_LEN = 60         # 🔁 longer output

X, y = [], []

for i in range(len(text) - SEQ_LEN):
    X.append([char_to_idx[c] for c in text[i:i+SEQ_LEN]])
    y.append(char_to_idx[text[i+SEQ_LEN]])

X = tf.one_hot(np.array(X), len(chars))
y = tf.one_hot(np.array(y), len(chars))

# ---------------- MODEL ----------------
model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(SEQ_LEN, len(chars))),  # 🔁 tanh + more units
    Dropout(0.3),                                                         # 🔁 new
    Dense(len(chars), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.005),   # 🔁 custom LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
model.fit(X, y, epochs=60, verbose=1)

# ---------------- TEMPERATURE SAMPLING (NEW) ----------------
def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# ---------------- TEXT GENERATION ----------------
start_seq = "Artificial intelligence "
generated = start_seq

for _ in range(GEN_LEN):
    seq = generated[-SEQ_LEN:]
    x = tf.one_hot([[char_to_idx[c] for c in seq]], len(chars))
    preds = model.predict(x, verbose=0)[0]
    next_idx = sample_with_temperature(preds)
    generated += idx_to_char[next_idx]

print("\nGenerated Text:")
print(generated)
