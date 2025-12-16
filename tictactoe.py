# TicTacToe_Friend_Final_Unique.py

import random

WIN = 1.0
LOSS = -1.0
DRAW = 0.4

class TicTacToeAI:
    def __init__(self):
        self.Q = {}
        self.lr = 0.25
        self.gamma = 0.85
        self.epsilon = 0.8
        self.epsilon_decay = 0.99

    # ---------- UTILITIES ----------
    def get_state(self, board):
        return tuple(board)

    def available_moves(self, board):
        return [i for i in range(9) if board[i] == 0]

    def check_win(self, b):
        wins = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        return any(sum(b[i] for i in w) == 3 for w in wins)

    def display(self, board):
        sym = {1:'X', -1:'O', 0:'_'}
        print()
        for i in range(0,9,3):
            print(sym[board[i]], sym[board[i+1]], sym[board[i+2]])

    # ---------- ACTION SELECTION ----------
    def choose_action(self, board, explore=True):
        state = self.get_state(board)
        moves = self.available_moves(board)

        if explore and random.random() < self.epsilon:
            return random.choice(moves)

        q_vals = [(self.Q.get((state,m), 0), m) for m in moves]
        return max(q_vals)[1]

    # ---------- EPISODE UPDATE ----------
    def update_episode(self, states, actions, reward):
        for s, a in zip(reversed(states), reversed(actions)):
            old = self.Q.get((s,a), 0)
            self.Q[(s,a)] = old + self.lr * (reward - old)
            reward *= self.gamma

    # ---------- TRAINING ----------
    def train(self, episodes=12000):
        wins = draws = losses = 0

        for _ in range(episodes):
            board = [0]*9
            states, actions = [], []

            while True:
                action = self.choose_action(board)
                states.append(self.get_state(board))
                actions.append(action)

                board[action] = -1  # AI move

                if self.check_win(board):
                    self.update_episode(states, actions, WIN)
                    wins += 1
                    break

                if not self.available_moves(board):
                    self.update_episode(states, actions, DRAW)
                    draws += 1
                    break

                # Human simulated move
                board[random.choice(self.available_moves(board))] = 1

                if self.check_win(board):
                    self.update_episode(states, actions, LOSS)
                    losses += 1
                    break

            self.epsilon *= self.epsilon_decay

        print("\nTraining Completed")
        print(f"Win Rate : {wins/episodes:.2f}")
        print(f"Draw Rate: {draws/episodes:.2f}")
        print(f"Loss Rate: {losses/episodes:.2f}")

    # ---------- HUMAN PLAY ----------
    def play(self):
        self.epsilon = 0  # no exploration
        board = [0]*9

        print("\nYou are X | Computer is O")
        self.display(board)

        while True:
            # ---- HUMAN MOVE ----
            try:
                r = int(input("Row (0-2): "))
                c = int(input("Col (0-2): "))
                pos = r*3 + c
                if pos not in self.available_moves(board):
                    print("Invalid move.")
                    continue
            except:
                print("Enter valid row & column.")
                continue

            board[pos] = 1
            self.display(board)

            if self.check_win(board):
                print("\n🎉 Game Over: You Win!")
                break

            if not self.available_moves(board):
                print("\n🤝 Game Over: Draw!")
                break

            # ---- COMPUTER MOVE ----
            ai_move = self.choose_action(board, explore=False)
            board[ai_move] = -1
            print("Computer move:")
            self.display(board)

            if self.check_win(board):
                print("\n💻 Game Over: Computer Wins!")
                break


# ---------- RUN ----------
ai = TicTacToeAI()
ai.train()

while True:
    ai.play()
    if input("Play again? (y/n): ").lower() != 'y':
        break
