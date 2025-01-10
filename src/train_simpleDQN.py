import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import os
import joblib
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


class HIVTreatmentDQN(nn.Module):
    def __init__(self):
        super(HIVTreatmentDQN, self).__init__()
        
        # Network layers
        self.layers = nn.Sequential(
            nn.Linear(6, 64),  # 6 state variables
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)   # 4 actions (no drugs, drug1, drug2, both)
        )
        
        # Initialize weights using Xavier/Glorot initialization
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # Ensure input is float tensor and on correct device
        if not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor):
            x = torch.FloatTensor(x)
        
        # Forward pass
        return self.layers(x)

class ProjectAgent:
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network architecture
        self.model = HIVTreatmentDQN().to(self.device)
        self.target_model = deepcopy(self.model)
        
        # Training parameters
        self.config = {
            'nb_actions': 4,
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'buffer_size': 100000,
            'epsilon_max': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay_period': 10000,
            'epsilon_delay_decay': 5000,
            'batch_size': 32,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 200,
            'update_target_tau': 0.005,
            'use_Huber_loss': True
        }
        
        # Initialize memory buffer
        self.memory = ReplayBuffer(self.config['buffer_size'], self.device)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.SmoothL1Loss() if self.config['use_Huber_loss'] else nn.MSELoss()
        
        # Training variables
        self.epsilon = self.config['epsilon_max']
        self.total_steps = 0
        
    def normalize_state(self, state):
        """Normalize state values to reasonable ranges and ensure float32 dtype"""
        norm_factors = np.array([1e6, 5e4, 3200.0, 1e2, 2.5e5, 3.532e5])
        state = state / norm_factors
        return torch.tensor(state, dtype=torch.float32)
        
    def act(self, state, use_random=False):
        """Select action using epsilon-greedy policy"""
        if use_random:
            return np.random.randint(self.config['nb_actions'])
            
        with torch.no_grad():
            state = self.normalize_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
            
    def gradient_step(self):
        if len(self.memory) > self.config['batch_size']:
            X, A, R, Y, D = self.memory.sample(self.config['batch_size'])
            # Normalize states
            X = self.normalize_state(X)
            Y = self.normalize_state(Y)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.config['gamma'])
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
    def update_target_network(self):
        """Update target network either by replacement or EMA"""
        if self.config['update_target_strategy'] == 'replace':
            if self.total_steps % self.config['update_target_freq'] == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        else:  # EMA update
            tau = self.config['update_target_tau']
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
    def update_epsilon(self):
        """Update exploration rate"""
        if self.total_steps > self.config['epsilon_delay_decay']:
            self.epsilon = max(
                self.config['epsilon_min'],
                self.epsilon - (self.config['epsilon_max'] - self.config['epsilon_min']) / self.config['epsilon_decay_period']
            )


    def train(self, env, max_episodes):
        """Train the DQN agent with tqdm progress bar."""
        episode_returns = []
        cumulative_reward = 0
        state, _ = env.reset()

        # Create a tqdm progress bar
        with tqdm(total=max_episodes, desc="Training Progress", unit="episode") as pbar:
            while len(episode_returns) < max_episodes:
                # Select action with epsilon-greedy policy
                action = self.act(state, use_random=np.random.rand() < self.epsilon)
                
                # Step in the environment
                next_state, reward, done, truncated, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                cumulative_reward += reward
                
                # Perform gradient updates
                for _ in range(self.config['gradient_steps']):
                    self.gradient_step()
                
                self.update_target_network()
                self.update_epsilon()
                self.total_steps += 1

                if done or truncated:
                    # Increment episode count
                    episode_returns.append(cumulative_reward)
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "Epsilon": f"{self.epsilon:.3f}",
                        "Return": f"{cumulative_reward:.1f}",
                        "Memory": len(self.memory)
                    })

                    # Print episode summary (optional)
                    print(f"Episode {len(episode_returns):3d}, Epsilon {self.epsilon:.3f}, "
                        f"Memory Size {len(self.memory):5d}, Episode Return {cumulative_reward:.1f}")

                    # Reset cumulative reward and environment
                    cumulative_reward = 0
                    state, _ = env.reset()
                else:
                    state = next_state

        return episode_returns


            
    def save(self, path):
        """Save the DQN model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, "saved_model.pt")
        
    def load(self):
        """Load the DQN model"""
        try:
            checkpoint = torch.load("saved_model.pt", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            print("Model loaded successfully")
        except:
            print("No saved model found. Starting from scratch.")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    def set_seeds(seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seeds()

    env = TimeLimit(HIVPatient(), max_episode_steps=200)
    agent = ProjectAgent()

    # Training configuration
    max_episodes = 500  # Number of episodes to train the model
    save_path = "saved_model.pt"  # Path to save the trained model

    # Train the agent
    print("Training the DQN agent...")
    episode_returns = agent.train(env, max_episodes)

    # Save the trained model
    print(f"Saving the trained model to {save_path}...")
    agent.save(save_path)

    print("Training complete!")
