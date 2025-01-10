import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import os
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import tqdm
import csv
import os
from datetime import datetime

class NStepPrioritizedReplayBuffer:
    """Implements prioritized experience replay with N-step returns.
    
    This buffer stores transitions and computes N-step returns on the fly.
    It combines three important concepts:
    1. Experience replay: storing and reusing past experiences
    2. Prioritized sampling: focusing on important transitions
    3. N-step returns: using multiple steps for better credit assignment
    """
    def __init__(self, capacity, device, n_step=3, gamma=0.995, alpha=0.6):
        self.capacity = int(capacity)
        self.device = device
        self.alpha = alpha        # Controls how much prioritization to use
        self.beta = 0.4          # Initial importance sampling weight
        self.beta_increment = 0.001  # How much to increase beta over time
        self.n_step = n_step     # Number of steps to look ahead
        self.gamma = gamma       # Discount factor
        
        # Main storage for transitions
        self.data = []           # Stores (state, action, n_step_reward, next_state, done)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        # Temporary buffer for computing n-step returns
        self.n_step_buffer = []  # Stores recent transitions for n-step computation
        self.index = 0           # Current position in the circular buffer
        
    def _get_n_step_info(self, n_step_buffer):
        """Computes the N-step return information from a sequence of transitions.
        
        Args:
            n_step_buffer: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            tuple: (n_step_reward, final_next_state, final_done)
            - n_step_reward: Sum of discounted rewards over n steps
            - final_next_state: The state after n steps
            - final_done: Whether any state in the sequence was terminal
        """
        # Get the last transition's next_state and done flag
        _, _, _, last_next_state, last_done = n_step_buffer[-1]
        
        # Calculate n-step discounted reward
        n_step_reward = 0
        for idx, (_, _, r, _, _) in enumerate(n_step_buffer):
            n_step_reward += (self.gamma ** idx) * r
            
        return n_step_reward, last_next_state, last_done
        
    def append(self, state, action, reward, next_state, done):
        """Adds a new transition to the buffer and computes n-step returns.
        
        This method:
        1. Adds the new transition to the n-step buffer
        2. If we have enough transitions, computes and stores an n-step transition
        3. Handles episode termination by storing remaining n-step transitions
        """
        # Add transition to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Not enough transitions for n-step return yet
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # Compute n-step return information
        n_reward, n_next_state, n_done = self._get_n_step_info(self.n_step_buffer)
        
        # Get initial state and action
        init_state, init_action = self.n_step_buffer[0][:2]
        
        # Store the n-step transition
        max_priority = np.max(self.priorities) if self.data else 1.0
        
        if len(self.data) < self.capacity:
            self.data.append(None)
            
        # Store (state, action, n_step_reward, n_step_next_state, n_done)
        self.data[self.index] = (init_state, init_action, n_reward, n_next_state, n_done)
        self.priorities[self.index] = max_priority
        self.index = (self.index + 1) % self.capacity
        
        # Remove oldest transition
        self.n_step_buffer.pop(0)
        
        # If episode terminated, process remaining transitions
        if done:
            while len(self.n_step_buffer) > 0:
                # Compute returns for shorter sequences
                n_reward, n_next_state, n_done = self._get_n_step_info(self.n_step_buffer)
                init_state, init_action = self.n_step_buffer[0][:2]
                
                # Store the transition
                if len(self.data) < self.capacity:
                    self.data.append(None)
                    
                self.data[self.index] = (init_state, init_action, n_reward, n_next_state, n_done)
                self.priorities[self.index] = max_priority
                self.index = (self.index + 1) % self.capacity
                
                self.n_step_buffer.pop(0)

    def sample(self, batch_size):
        """Samples a batch of transitions using prioritized sampling.
        
        Higher priority transitions are more likely to be sampled.
        Returns importance sampling weights to correct for the bias.
        """
        if len(self.data) == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[:len(self.data)]
            
        # Convert priorities to probabilities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices and compute importance sampling weights
        indices = np.random.choice(len(self.data), batch_size, p=probs)
        
        # Compute importance sampling weights
        self.beta = np.min([1., self.beta + self.beta_increment])
        weights = (len(self.data) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize weights
        
        # Get the sampled transitions
        samples = [self.data[idx] for idx in indices]
        
        # Convert to PyTorch tensors and move to device
        batch = list(map(lambda x: torch.tensor(np.array(x)).to(self.device), zip(*samples)))
        weights = torch.FloatTensor(weights).to(self.device)
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Updates priorities of sampled transitions using latest TD-errors.
        
        Args:
            indices: List of indices of sampled transitions
            td_errors: List of TD-errors for the sampled transitions
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6  # Small constant prevents zero priority
            
    def __len__(self):
        """Returns current length of the replay buffer"""
        return len(self.data)

class HIVTreatmentDQN(nn.Module):
    """Enhanced DQN architecture with normalization and regularization"""
    def __init__(self):
        super().__init__()
        
        # Initial layer processes raw state
        self.input_layer = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),  # Normalization helps with different scales
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate paths for immune system and viral dynamics
        self.immune_path = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        self.viral_path = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Combine paths
        self.combine = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        # Initialize weights using Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        if not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        
        x = self.input_layer(x)
        immune = self.immune_path(x)
        viral = self.viral_path(x)
        combined = torch.cat([immune, viral], dim=-1)
        return self.combine(combined)

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
            'n_step': 10,
            'learning_rate': 3e-4,
            'gamma': 0.995,
            'buffer_size': 100000,
            'epsilon_max': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay_period': 50000,
            'epsilon_delay_decay': 7000,
            'batch_size': 128,
            'gradient_steps': 4,
            'update_target_tau': 0.001
        }
        
        # Initialize replay buffer with prioritization
        self.memory = NStepPrioritizedReplayBuffer(self.config['buffer_size'], self.device, n_step=self.config["n_step"])
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # Training variables
        self.epsilon = self.config['epsilon_max']
        self.total_steps = 0
        
        # State normalization statistics
        self.state_mean = None
        self.state_std = None
        
    def normalize_state(self, state):
        """Normalize state using log transformation and running statistics"""
        # Handle both single states and batches
        is_single = len(state.shape) == 1
        if is_single:
            state = state[np.newaxis, :]
            
        # Log transform to handle large value ranges
        state = np.log1p(state)
        
        # Initialize or update running statistics
        if self.state_mean is None:
            self.state_mean = np.mean(state, axis=0)  # Shape: (6,)
            self.state_std = np.std(state, axis=0)    # Shape: (6,)
        else:
            # Update running statistics during training
            self.state_mean = 0.99 * self.state_mean + 0.01 * np.mean(state, axis=0)
            self.state_std = 0.99 * self.state_std + 0.01 * np.std(state, axis=0)
        
        # Normalize
        normalized = (state - self.state_mean) / (self.state_std + 1e-8)
        
        # Return same shape as input
        return normalized[0] if is_single else normalized
        
    def act(self, state, use_random=False):
        """Modified action selection with intelligent exploration"""
        if use_random:
            # Bias initial exploration:
            # Higher chance of no drugs (0) or single drug (1,2) vs both drugs (3)
            # This aligns with paper's insight about periodic treatment interruption
            probs = [0.4, 0.2, 0.2, 0.2]  # 40% chance no drugs, 20% each other action
            return np.random.choice(self.config['nb_actions'], p=probs)
            
        with torch.no_grad():
            state = self.normalize_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
            
    def gradient_step(self):
        """Performs a gradient step using n-step returns"""
        if len(self.memory) > self.config['batch_size']:
            # Sample batch with priorities
            batch, indices, weights = self.memory.sample(self.config['batch_size'])
            states, actions, n_step_rewards, next_states, dones = batch
            dones = dones.float()
            
            # Normalize states
            states = torch.tensor(self.normalize_state(states.cpu().numpy())).float().to(self.device)
            next_states = torch.tensor(self.normalize_state(next_states.cpu().numpy())).float().to(self.device)
            
            # Current Q-values for the actions taken
            current_q_values = self.model(states).gather(1, actions.long().unsqueeze(1))
            
            # Next state values (from target network)
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0]
                
                # For n-step returns, we need to adjust the bootstrap term
                # The reward already includes n-1 steps, so we only need gamma^n for the bootstrap
                bootstrap = (self.config['gamma'] ** self.config['n_step']) * next_q_values
                
                # Target is n-step reward plus bootstrap term (if not done)
                targets = n_step_rewards + (1 - dones) * bootstrap
            
            # Compute loss with importance sampling weights
            loss = (weights * (targets.unsqueeze(1) - current_q_values) ** 2).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update priorities
            td_errors = (targets - current_q_values.squeeze()).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)
            
    def update_target_network(self):
        """Update target network using soft updates"""
        tau = self.config['update_target_tau']
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
    def update_epsilon(self):
        """Update exploration rate with delayed decay"""
        if self.total_steps > self.config['epsilon_delay_decay']:
            self.epsilon = max(
                self.config['epsilon_min'],
                self.epsilon - (self.config['epsilon_max'] - self.config['epsilon_min']) 
                / self.config['epsilon_decay_period']
            )

    def train(self, env, max_episodes):
        """Train the agent with progress tracking and monitoring"""
        episode_returns = []
        running_reward = 0
        best_return = float('-inf')

        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create a new log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/training_log_{timestamp}.csv'
        
        # Initialize the CSV writer
        first_episode = True
        csv_file = open(log_file, 'w', newline='')
        csv_writer = None
        
        with tqdm(total=max_episodes, desc="Training Progress") as pbar:
            for episode in range(max_episodes):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    # Select and perform action
                    action = self.act(state, use_random=np.random.random() < self.epsilon)
                    next_state, reward, done, truncated, _ = env.step(action)
                    
                    # Store transition
                    self.memory.append(state, action, reward, next_state, done)
                    
                    # Multiple gradient steps per environment step
                    for _ in range(self.config['gradient_steps']):
                        self.gradient_step()
                    
                    # Update target network
                    self.update_target_network()
                    
                    # Update exploration rate
                    self.update_epsilon()
                    
                    episode_reward += reward
                    state = next_state
                    self.total_steps += 1
                    
                    if truncated:
                        done = True
                
                # Update statistics and logging
                episode_returns.append(episode_reward)
                running_reward = 0.05 * episode_reward + 0.95 * running_reward
                
                # Save best model
                if running_reward > best_return:
                    best_return = running_reward
                    self.save('best_model.pt')
                
                # Collect and log statistics
                stats = self.collect_training_stats(episode, episode_reward, running_reward)
                
                # Initialize CSV writer with headers on first episode
                if first_episode:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=stats.keys())
                    csv_writer.writeheader()
                    first_episode = False
                
                # Write statistics to CSV
                csv_writer.writerow(stats)
                csv_file.flush()  # Ensure data is written to file

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f'{episode_reward:.0f}',
                    'running_reward': f'{running_reward:.0f}',
                    'epsilon': f'{self.epsilon:.2f}'
                })
        csv_file.close()
        return episode_returns
            
    def save(self, path):
        """Save the agent's state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'state_mean': self.state_mean,
            'state_std': self.state_std
        }, path)
        
    def load(self):
        """Load the agent's state"""
        try:
            checkpoint = torch.load("saved_model.pt", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            self.state_mean = checkpoint.get('state_mean', None)
            self.state_std = checkpoint.get('state_std', None)
            print("Model loaded successfully")
        except:
            print("No saved model found. Starting from scratch.")
    
    def collect_training_stats(self, episode, episode_reward, running_reward):
        """Gather all relevant training statistics into a dictionary"""
        # Calculate model statistics
        param_norm = 0.0
        grad_norm = 0.0
        for param in self.model.parameters():
            param_norm += param.data.norm(2).item() ** 2
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        param_norm = np.sqrt(param_norm)
        grad_norm = np.sqrt(grad_norm)

        # Calculate replay buffer statistics
        if len(self.memory) > 0:
            priority_mean = np.mean(self.memory.priorities[:len(self.memory)])
            priority_std = np.std(self.memory.priorities[:len(self.memory)])
        else:
            priority_mean = 0.0
            priority_std = 0.0

        # Compile all statistics
        stats = {
            'total_steps': self.total_steps,
            'episode': episode,
            'epsilon': self.epsilon,
            'episode_reward': episode_reward,
            'running_reward': running_reward,
            'memory_size': len(self.memory),
            'priority_mean': priority_mean,
            'priority_std': priority_std,
            'param_norm': param_norm,
            'grad_norm': grad_norm,
            'beta': self.memory.beta,
            'state_mean_1': float(self.state_mean[0]),
            'state_mean_2': float(self.state_mean[1]),
            'state_mean_3': float(self.state_mean[2]),
            'state_mean_4': float(self.state_mean[3]),
            'state_mean_5': float(self.state_mean[4]),
            'state_mean_6': float(self.state_mean[5]),
            'state_std_1': float(self.state_std[0]),
            'state_std_2': float(self.state_std[1]),
            'state_std_3': float(self.state_std[2]),
            'state_std_4': float(self.state_std[3]),
            'state_std_5': float(self.state_std[4]),
            'state_std_6': float(self.state_std[5])
        }
        return stats

if __name__ == "__main__":
    # Set random seeds for reproducibility
    def set_seeds(seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    set_seeds()
    
    # Create environment and agent
    env = TimeLimit(HIVPatient(), max_episode_steps=200)
    agent = ProjectAgent()
    
    # Training configuration
    max_episodes = 500
    save_path = "saved_model.pt"
    
    # Train the agent
    print("Training the DQN agent...")
    episode_returns = agent.train(env, max_episodes)
    
    # Save the trained model
    print(f"Saving the trained model to {save_path}...")
    agent.save(save_path)
    
    print("Training complete!")