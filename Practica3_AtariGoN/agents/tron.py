import random
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from collections import deque
import numpy as np

from atarigon.api import Goshi, Goban, Ten

class DQN(nn.module):
    def _init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
    
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)

class TronGoshi(Goshi):
    """Player that makes random moves.

    You'll never know what it's going to do next!
    """

    # Hiperparámetros
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10
    replay_memory_size = 10000
    mini_batch_size = 64

    # Cosas de la red neuronal
    loss_fn = nn.MSELoss()
    goban_size = 9
    num_states = goban_size*goban_size
    num_actions = goban_size*goban_size

    epsilon = 1
    memory = ReplayMemory(replay_memory_size)

    # Creamos las redes de política y objetivo
    policy_dqn = DQN(in_states=num_states, h1_nodes= num_states, out_actions = num_actions)
    target_dqn = DQN(in_states=num_states, h1_nodes= num_states, out_actions = num_actions)

    target_dqn.load_state_dict(policy_dqn.state_dict())

    optimizer = torch.optim.Adam(policy_dqn.parameters())

    # Para actualizar la red objetivo
    step_count = 0

    def __init__(self):
        """Initializes the player with the given name."""
        super().__init__(f'Tron')        

    def decide(self, goban: 'Goban') -> Optional[Ten]:
        """Gets a random empty position in the board.

        :param goban: The current observation of the game.
        :return: The next move as a (row, col) tuple.
        """

        state = self.goban_to_state(goban)

        # Finds all the empty positions in the observation
        empty_positions = [
            Ten(row, col)
            for row in range(len(goban.ban))
            for col in range(len(goban.ban[row]))
            if goban.ban[row][col] is None
        ]

        # Chooses a random valid empty position
        random.shuffle(empty_positions)
        for ten in empty_positions:
            if goban.ban[ten.row][ten.col] is None:
                random_position = ten.row*self.goban_size + ten.col

        # Action es un valor de 0 a n, con n el total de casillas en el tablero
        # Al final se convierte a una coordenada del tablero con la clase Ten.
        if random.random() < self.epsilon:
            action = random_position
        else:
            action = self.policy_dqn(state).argmax().item()
        
        new_state = self.get_new_state(goban, action)
        reward = self.reward_function(goban, ten)
        terminated = goban.seichō(action, self) or goban.jishi(action, self) # Añadir la condición si es el único jugador, se terminó
        self.memory.append((state, action, new_state, reward, terminated))

        step_count+=1

        if len(self.memory) > self.mini_batch_size: # Revisar si también debería de ponerse la condicional de haber obtenido al menos alguna recompensa
            mini_batch = self.memory.sample(self.mini_batch_size)
            self.optimize(mini_batch)

            # Decaemiento de épsilon
            self.epsilon = self.epsilon*0.995

            # Copia de la red de política después de tantos pasos
            if step_count > self.network_sync_rate:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                step_count = 0

        action = self.output_to_ten(action)

        if goban.seichō(action, self):
            action = self.output_to_ten(random_position)
        return action # Tiene que regresar un Ten, pero revisar para las otras funciones

    # Actualización de q values
    def optimize(self, mini_batch):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                # El jugador terminó el juego, ya sea por suicidio, colocar mal una pieza o ser el último
                # Si terminó, el q value objetivo debe ser igual a la recompensa
                target = torch.FloatTensor([reward])
            else:
                # Calculamos el valor q
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * self.target_dqn(new_state).max()
                    )

            # Obtenemos los valores q de la política
            current_q = self.policy_dqn(state)
            current_q_list.append(current_q)

            # Obtenemos los valos q del objetivo
            target_q = self.target_dqn(state)
            # Ajuste de la acción al objetivo que se calculó
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        torch.save(self.policy_dqn.state_dict(), "../tron_dql.pt")

    # Función de recompensa
    def reward_function(self, goban: 'Goban', ten: 'Ten'):
        reward = 0.0
        return 1
    
    def get_new_state(self, goban: 'Goban', ten: 'Ten'):
        board = self.goban_to_state(goban)
        board[ten.row*self.goban_size + ten.col]
        return board

    # Pasamos del formato de ban de goban a uno donde sólo haya números
    def goban_to_state(self, goban: 'Goban'):
        board = np.zeros(goban.size*goban.size)
        for i, row in enumerate(goban.ban):
            for j, p in enumerate(row):
                if p is not None:
                    board[i*goban.size + j] = goban.stone_colors[p]
        return board
    
    # Salida a coordenadas del tablero
    def output_to_ten(self, output):
        row, column = divmod(output, self.goban_size) 
        return Ten(row, column)