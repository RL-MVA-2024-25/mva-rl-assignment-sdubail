import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import tqdm
import torch.nn.functional as F
import math

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
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
    
    def save_buffer(self, path):
        """Save the replay buffer to disk"""
        buffer_data = {
            'capacity': self.capacity,
            'data': self.data,
            'index': self.index
        }
        torch.save(buffer_data, path)
        
    def load_buffer(self, path):
        """Load the replay buffer from disk"""
        buffer_data = torch.load(path)
        self.capacity = buffer_data['capacity']
        self.data = buffer_data['data']
        self.index = buffer_data['index']

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5, factorized=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorized = factorized
        self.std_init = std_init
        
        # Learnable parameters (μ)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        # Noise scaling parameters (σ)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise parameters
        if factorized:
            self.register_buffer('weight_epsilon_input', torch.empty(in_features))
            self.register_buffer('weight_epsilon_output', torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
        else:
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
            
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        # Initialize μ using Glorot initialization
        std = math.sqrt(3 / self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.bias_mu.data.uniform_(-std, std)
        
        # Initialize σ
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())
        
    def reset_noise(self):
        if self.factorized:
            epsilon_input = self._scale_noise(self.in_features)
            epsilon_output = self._scale_noise(self.out_features)
            self.weight_epsilon_input.copy_(epsilon_input)
            self.weight_epsilon_output.copy_(epsilon_output)
            self.bias_epsilon.copy_(self._scale_noise(self.out_features))
        else:
            self.weight_epsilon.copy_(torch.randn(self.out_features, self.in_features))
            self.bias_epsilon.copy_(torch.randn(self.out_features))
            
    def forward(self, x):
        if self.training:
            if self.factorized:
                # Factorized Gaussian noise
                weight = self.weight_mu + self.weight_sigma * torch.ger(
                    self.weight_epsilon_output, self.weight_epsilon_input)
            else:
                # Independent Gaussian noise
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class NoisyHIVTreatmentDQN(nn.Module):
    def __init__(self):
        super(NoisyHIVTreatmentDQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(6, 64),  # First layer remains standard
            nn.ReLU(),
            NoisyLinear(64, 64, std_init=0.5),  # Replace standard linear layers with noisy ones
            nn.ReLU(),
            NoisyLinear(64, 4, std_init=0.5)
        )
        
        # Initialize non-noisy layers
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        if not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor):
            x = torch.FloatTensor(x)
        return self.layers(x)
        
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class ProjectAgent:
    def __init__(self, N_explo=50, T_explo=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Exploration parameters
        self.N_explo = N_explo  # Number of exploration episodes
        self.T_explo = T_explo  # Steps before changing random policy
        
        # Network architecture
        self.model = NoisyHIVTreatmentDQN().to(self.device)
        self.target_model = deepcopy(self.model)
        self.training = False
        self.config = {
            'nb_actions': 4,
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'buffer_size': 100000,
            'batch_size': 256,
            'gradient_steps': 2,
            'update_target_strategy': 'ema',
            'update_target_freq': 200,
            'update_target_tau': 0.005,
            'use_Huber_loss': True
        }
        
        self.memory = ReplayBuffer(self.config['buffer_size'], self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.SmoothL1Loss() if self.config['use_Huber_loss'] else nn.MSELoss()
        self.total_steps = 0
        self.reward_scale = 1e-6
        
    def normalize_state(self, state):
        """Log-normalize state values to handle varying magnitudes better"""
        norm_factors = np.array([1e6, 5e4, 3200.0, 1e2, 2.5e5, 3.532e5])
        epsilon = 1e-8
        normalized = np.log1p(state + epsilon) / np.log1p(norm_factors + epsilon)
        return torch.tensor(normalized, dtype=torch.float32)
        
    def get_sigma_magnitude(self):
        """Get average magnitude of sigma parameters for all noisy layers"""
        total_sigma = 0
        count = 0
        for module in self.model.modules():
            if isinstance(module, NoisyLinear):
                total_sigma += module.weight_sigma.data.abs().mean().item()
                total_sigma += module.bias_sigma.data.abs().mean().item()
                count += 2
        return total_sigma / count if count > 0 else 0
    
    def get_network_stats(self):
        """Get statistics about mu and sigma parameters for all noisy layers"""
        stats = {
            'weight_mu': [],
            'bias_mu': [],
            'weight_sigma': [],
            'bias_sigma': []
        }
        
        for module in self.model.modules():
            if isinstance(module, NoisyLinear):
                # Get mean values (μ)
                stats['weight_mu'].append(module.weight_mu.data.abs().mean().item())
                stats['bias_mu'].append(module.bias_mu.data.abs().mean().item())
                
                # Get noise scale values (σ)
                stats['weight_sigma'].append(module.weight_sigma.data.abs().mean().item())
                stats['bias_sigma'].append(module.bias_sigma.data.abs().mean().item())
        
        # Average across layers
        return {k: sum(v)/len(v) if v else 0 for k, v in stats.items()}
    
    def get_average_q_values(self, state):
        """Get average Q-values for each action in the current state"""
        with torch.no_grad():
            state = self.normalize_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Sample multiple times with different noise to get average Q-values
            n_samples = 10
            q_values = []
            for _ in range(n_samples):
                self.model.reset_noise()
                q_values.append(self.model(state_tensor).cpu().numpy())
            
            return np.mean(q_values, axis=0)[0]

    def gradient_step(self):
        if len(self.memory) > self.config['batch_size']:
            # Sample fresh noise for training
            self.model.reset_noise()
            self.target_model.reset_noise()
            
            # Sample and prepare batch
            X, A, R, Y, D = self.memory.sample(self.config['batch_size'])
            X = self.normalize_state(X)
            Y = self.normalize_state(Y)
            
            # Double DQN: Use online network to select action, target network to evaluate it
            with torch.no_grad():
                # Get actions from online network
                next_actions = self.model(Y).argmax(1)
                # Get Q-values from target network
                next_q_values = self.target_model(Y)
                # Use the actions from online network to select Q-values from target network
                next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = R + (1-D) * self.config['gamma'] * next_q_values # scaling the reward.
                
            # Current Q-values
            QXA = self.model(X).gather(1, A.long().unsqueeze(1))
            
            # Loss calculation
            loss = self.criterion(QXA, target.unsqueeze(1))
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def act(self, state, use_random=False):
        """Select action with fresh noise sample"""
        with torch.no_grad():
            if use_random:
                return random.randrange(self.config['nb_actions'])
            
            if not self.training:
                self.model.eval()
            else:
                self.model.train()
                
            # Sample fresh noise for action selection
            self.model.reset_noise()
            
            state = self.normalize_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def update_target_network(self):
        """Update target network"""
        if self.config['update_target_strategy'] == 'replace':
            if self.total_steps % self.config['update_target_freq'] == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        else:  # EMA update
            tau = self.config['update_target_tau']
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train(self, env, max_episodes):
        """Train the agent with initial exploration phase"""
        episode_returns = []
        cumulative_reward = 0
        state, _ = env.reset()
        action_counts = np.zeros(4)
        
        # Exploration phase tracking
        steps_since_policy_change = 0
        current_exploration_policy = np.random.randint(0, 4)  # Initial random policy

        with tqdm(total=max_episodes, desc="Training Progress", unit="episode") as pbar:
            while len(episode_returns) < max_episodes:
                # Determine if we're in exploration phase
                in_exploration = len(episode_returns) < self.N_explo
                
                # Update exploration policy if needed
                if in_exploration and steps_since_policy_change >= self.T_explo:
                    current_exploration_policy = np.random.randint(0, 4)
                    steps_since_policy_change = 0
                
                # Select action
                if in_exploration:
                    action = current_exploration_policy  # Use current fixed random policy
                    steps_since_policy_change += 1
                else:
                    action = self.act(state)  # Use NoisyNet policy
                
                action_counts[action] += 1
                
                # Environment step
                next_state, reward, done, truncated, _ = env.step(action)
                scaled_reward = self.reward_scale * reward
                self.memory.append(state, action, scaled_reward, next_state, done)
                cumulative_reward += reward
                
                # Training steps (even during exploration to learn from random experience)
                for _ in range(self.config['gradient_steps']):
                    self.gradient_step()
                self.update_target_network()

                self.total_steps += 1

                if done or truncated:
                    episode_returns.append(cumulative_reward)
                    total_actions = sum(action_counts)
                    action_probs = action_counts / total_actions if total_actions > 0 else action_counts
                    net_stats = self.get_network_stats()
                    avg_q_values = self.get_average_q_values(state)
                    pbar.update(1)
                    pbar.set_postfix({
                        "Return": f"{cumulative_reward:.1f}",
                        "Memory": len(self.memory),
                        "Phase": "Exploration" if in_exploration else "Training"
                    })

                    print(f"Episode {len(episode_returns):3d}, "
                        f"Memory Size {len(self.memory):5d}, "
                        f"Episode Return {cumulative_reward:.1f}, "
                        f"Action probs {action_probs}, "
                        f"μ_w: {net_stats['weight_mu']:.3f}, "
                        f"μ_b: {net_stats['bias_mu']:.3f}, "
                        f"σ_w: {net_stats['weight_sigma']:.3f}, "
                        f"σ_b: {net_stats['bias_sigma']:.3f}, "
                        f"Q-values: {avg_q_values}, "
                        f"Phase: {'Exploration' if in_exploration else 'Training'}")

                    cumulative_reward = 0
                    action_counts = np.zeros(4)
                    state, _ = env.reset()

                    # Save checkpoint every 500 episodes
                    if len(episode_returns) % 500 == 0:
                        checkpoint_path = f"saved_model_noisynet_500buffer_{len(episode_returns)}.pt"
                        self.save(checkpoint_path)
                        print(f"Saved checkpoint at episode {len(episode_returns)}")

                else:
                    state = next_state

        return episode_returns
            
    def save(self, path):
        """Save the agent's state"""
        # Save model, optimizer state and config
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        
        # Save replay buffer to a separate file
        buffer_path = path.replace('.pt', '_buffer.pt')
        self.memory.save_buffer(buffer_path)
        print(f"Model saved to {path} and buffer saved to {buffer_path}")
        
    def load(self):
        """Load the agent's state"""
        try:
            # Load model and optimizer state
            checkpoint = torch.load("saved_model.pt", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            
            # Try to load replay buffer if it exists
            buffer_path = "saved_model_buffer.pt"
            try:
                self.memory.load_buffer(buffer_path)
                print(f"Model and replay buffer loaded successfully from {buffer_path}")
            except FileNotFoundError:
                print("Model loaded successfully, but no replay buffer found")
            except Exception as e:
                print(f"Model loaded but error loading buffer: {str(e)}")
        except FileNotFoundError:
            print("No saved model found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

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
    agent = ProjectAgent(N_explo=30, T_explo=10)
    agent.training = True
    # Training configuration
    max_episodes = 500
    save_path = "saved_model.pt"

    # Train the agent
    print("Training the agent...")
    episode_returns = agent.train(env, max_episodes)

    # Save the trained model
    print(f"Saving the trained model to {save_path}...")
    agent.save(save_path)

    print("Training complete!")