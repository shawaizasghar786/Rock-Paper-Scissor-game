import random
from collections import Counter

class SmartAI:
    def __init__(self):
        self.history=[]
        self.transitions = {
            'rock': {'rock': 0, 'paper': 0, 'scissors': 0},
            'paper': {'rock': 0, 'paper': 0, 'scissors': 0},
            'scissors': {'rock': 0, 'paper': 0, 'scissors': 0}
        }
        self.last_move=None

    def update(self, player_move):
        self.history.appent(player_move)
        if self.last_move:
            self.transitions[self.last_move][player_move]+=1
            self.last_move=player_move

    def predict(self):
        if not self.last_move:
            return random.choice(['rock', 'paper', 'scissors'])
        next_move=self.transitions[self.last_move]
        predicted=max(next_move, key=next_move.get)
        return self.counter(predicted)
    def counter(self, move):
        return {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}[move]
