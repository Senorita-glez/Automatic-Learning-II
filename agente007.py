import gym
import gym_go
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


def build_model(state_shape, action_size):
    model = Sequential()
    model.add(Flatten(input_shape=state_shape))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model(state_shape, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = np.random.choice(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def train_agent():
    env = gym.make('gym_go:go19-v0')
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_shape, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_shape[0], state_shape[1], state_shape[2]])
        done = False
        time = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_shape[0], state_shape[1], state_shape[2]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            time += 1

            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % 50 == 0:
            agent.save(f"weights_{e}.hdf5")


def calculate_reward(board, player):
    """
    Calcula la recompensa en función de las piedras capturadas.
    
    Args:
        board (np.array): Estado actual del tablero.
        player (int): Jugador actual (1 para negro, -1 para blanco).

    Returns:
        int: Recompensa basada en las piedras capturadas.
    """
    def count_captured_stones(board, player):
        captured_stones = 0
        opponent = -player
        
        visited = np.zeros_like(board)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] == opponent and not visited[i, j]:
                    stones, is_captured = dfs(board, i, j, visited, opponent)
                    if is_captured:
                        captured_stones += stones
        return captured_stones

    def dfs(board, x, y, visited, player):
        if x < 0 or y < 0 or x >= board.shape[0] or y >= board.shape[1]:
            return 0, False
        if visited[x, y] or board[x, y] != player:
            return 0, True
        visited[x, y] = 1
        stones = 1
        is_captured = True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            s, c = dfs(board, nx, ny, visited, player)
            stones += s
            is_captured = is_captured and c
        return stones, is_captured
    
    return count_captured_stones(board, player)

# --------------------------------------------

import numpy as np

def calcular_coeficiente_de_peligro(tablero, jugador):
    # Asumiendo que tablero es una matriz de NxN donde:
    # 0 representa una casilla vacía
    # 1 representa una ficha del jugador 1
    # 2 representa una ficha del jugador 2
    
    n = len(tablero)
    coeficiente_de_peligro = np.zeros((n, n))
    
    # Define los coeficientes
    coef_amigo = 1  # Coeficiente para una ficha amiga
    coef_enemigo = -1  # Coeficiente para una ficha enemiga
    
    # Asigna el coeficiente de acuerdo al jugador
    if jugador == 1:
        amigo = 1
        enemigo = 2
    else:
        amigo = 2
        enemigo = 1
    
    # Función auxiliar para calcular el coeficiente de una casilla
    def coeficiente_casilla(x, y):
        coef = 0
        # Coordenadas de los movimientos posibles (arriba, abajo, izquierda, derecha)
        movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in movimientos:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:  # Verifica que las coordenadas estén dentro del tablero
                if tablero[nx][ny] == amigo:
                    coef += coef_amigo
                elif tablero[nx][ny] == enemigo:
                    coef += coef_enemigo
        return coef
    
    # Calcula el coeficiente de peligro para cada casilla del tablero
    for i in range(n):
        for j in range(n):
            coeficiente_de_peligro[i][j] = coeficiente_casilla(i, j)
    
    # Normaliza los coeficientes a un rango de -4 a +4
    coeficiente_de_peligro = np.clip(coeficiente_de_peligro, -4, 4)
    
    return coeficiente_de_peligro

# Ejemplo de uso
tablero = [
    [0, 1, 0, 0],
    [2, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 2]
]

jugador = 1
coef_peligro = calcular_coeficiente_de_peligro(tablero, jugador)
print(coef_peligro)

