import random

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        # Saves a session
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        # Sample a session
        return random.sample(self.memory, 1)

    def __len__(self):
        return len(self.memory)