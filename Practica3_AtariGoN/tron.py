import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
from atarigon.api import Goshi, Goban, Ten
from typing import Optional


class TronGoshi(Goshi):
    """Player that learns using reinforcement learning algorithm: Deep Q Learning.

    Hopefully it will beat them all.
    """

    def __init__(self, epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995, gamma: float = 0.95, learning_rate: float = 0.001):
        """Initializes the player with the given parameters."""

        super().__init__(f'Tron')
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.replay_buffer = deque(maxlen=2000)

    def build_model(self, goban: 'Goban'):
        """Builds the neural network model."""
        model = Sequential([
            Dense(64, input_shape=(len(goban.ban) * len(goban.ban[0]),), activation='relu'),
            Dense(64, activation='relu'),
            Dense(len(goban.ban) * len(goban.ban[0]), activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def decide(self, goban: 'Goban') -> Optional[Ten]:
        """Selects the action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            # Explore: Choose a random empty position
            empty_positions = [
                Ten(row, col)
                for row in range(len(goban.ban))
                for col in range(len(goban.ban[row]))
                if goban.ban[row][col] is None
            ]
            return random.choice(empty_positions)
        else:
            # Exploit: Escogemos la acción con el valor Q más grande.
            # Obtenemos las piedras colocadas en forma de un arreglo unidimensional para alimentarlo a la nn.
            state = np.array([1 if cell is None else 0 for row in goban.ban for cell in row]) # Seguramente sea necesario identificar qué pidras son de quién
            state = state.reshape(1, -1)
            q_values = self.model.predict(state)
            q_values = q_values.reshape(len(goban.ban), len(goban.ban[0])) # Va a ser necesario regresar Ten, las coordenadas de la jugada.
            max_q_value = np.max(q_values)
            max_indices = np.where(q_values == max_q_value)
            max_indices = list(zip(max_indices[0], max_indices[1]))
            return random.choice(max_indices)

    def remember(self, state, action, reward, next_state, done):
        """Stores the experience in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size, goban: 'Goban'):
        """Trains the neural network using experiences from the replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.array(next_state)
                next_state = next_state.reshape(1, -1)
                target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(np.array(state).reshape(1, -1))
            target_f[0][action[0] * len(goban.ban[0]) + action[1]] = target
            self.model.fit(np.array(state).reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
