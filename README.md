
Deep Learning & Reinforcement Learning Assignment

| Field            | Details                                  |
| ---------------- | ---------------------------------------- |
|   Name           | Yogeshwaran P                            |
|   USN            | 1CD22AI063                               |
|   Subject        | Deep Learning and Reinforcement Learning |
|  Subject Code    | BAI701                                   |

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1️⃣ AlexNet.py — CNN Image Classification
🔹 Original Code

Standard AlexNet-style CNN

ReLU activation

Large kernels (11×11)

Flatten + Dense layers

No explicit architectural optimization

🔹 Modified Code (Friend Version – UNIQUE)

Exact Changes Made

Replaced ReLU with LeakyReLU

Added Batch Normalization after every convolution layer

Changed first convolution kernel size 11×11 → 7×7

Replaced Flatten with GlobalAveragePooling

Reduced dense layer dependency

🔹 Why These Changes Are Effective

Prevents vanishing gradient issues

Faster and more stable convergence

Fewer parameters → lower memory usage

Model summary becomes structurally different

✅ Uniqueness: Architecture-level change (not just tuning values)

2️⃣ TicTacToe.py — Reinforcement Learning Game
🔹 Original Code

Incorrect player symbol initialization

Weak reward structure

High randomness even after training

Poor input validation

Game sometimes continued after win/draw

🔹 Modified Code

Exact Changes Made

Fixed player symbol: 20 → 1

Improved reward system:

Win = +1

Loss = −1

Draw = +0.3

Reduced exploration rate: 0.3 → 0.1

Increased learning rate: 0.2 → 0.3

Reduced training rounds: 50000 → 20000

Enhanced board display (X O _ format)

Added input validation

Fixed game termination logic

🔹 Why These Changes Are Effective

Faster convergence

Smarter AI decisions

Stable gameplay

Better user interaction

3️⃣ RNN.py — Character-Level Text Generation
🔹 Original Code

Short training text

ReLU activation

Argmax-based text generation

No regularization

Limited creativity

🔹 Modified Code

Exact Changes Made

Changed training text to a technical sentence

Increased sequence length: 5 → 6

Changed activation: ReLU → tanh

Increased hidden units: 64

Added Dropout (0.3)

Custom Adam learning rate

Introduced temperature-based probabilistic sampling

Increased generated text length

🔹 Why These Changes Are Effective

Stable RNN training

Less repetitive text

More natural and creative output

Output becomes non-deterministic

✅ Uniqueness: Temperature-based sampling (advanced concept)

4️⃣ LSTM.py — Time Series Forecasting (Airline Passengers)
🔹 Original Code

Local dataset dependency

Single LSTM layer

Incorrect input shape

Batch size = 1

No validation or early stopping

🔹 Modified Code

Exact Changes Made

Dataset loaded from GitHub (portable)

Increased sequence length: 10 → 12

Corrected input shape (TIME_STEPS, 1)

Introduced stacked LSTM layers

Added Dropout

Added EarlyStopping

Improved batch size: 1 → 16

Added MAE along with RMSE

Improved prediction visualization

🔹 Why These Changes Are Effective

Better temporal modeling

Reduced overfitting

Faster and stable training

Improved forecasting accuracy

5️⃣ DeepReinforcementLearning.py — Q-Learning on Graph
🔹 Original Code

Sparse rewards (only goal reward)

Unstable Q-update rule

Pure random exploration

Complex environment heuristics (police/drugs)

Noisy learning curve

🔹 Modified Code

Exact Changes Made

Added step penalty (−1)

Used standard Q-learning update rule

Introduced learning rate (α)

Increased discount factor (γ)

Implemented epsilon-greedy exploration with decay

Tracked average Q-value instead of sum

Removed environment-specific heuristics

🔹 Why These Changes Are Effective

Encourages shortest path

Stable convergence

Smooth learning curve

Cleaner, algorithm-focused logic
