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

class NStepPrioritizedBuffer:
    """Implements prioritized experience replay with N-step returns.
    
    Combines:
    1. Experience replay for efficient learning from past experiences
    2. Prioritized sampling to focus on important transitions
    3. N-step returns for better credit assignment
    4. Numerically stable operations to prevent NaN errors
    """
    def __init__(self, capacity, device, n_step=3, gamma=0.99, alpha=0.6, 
                 beta_start=0.4, total_timesteps=100000):
        self.capacity = int(capacity)
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        
        # Beta annealing setup
        self.beta_start = beta_start
        self.beta_end = 1.0
        self.total_timesteps = total_timesteps
        self.beta = beta_start
        # Set increment to reach beta_end by total_timesteps
        self.beta_increment = (self.beta_end - self.beta_start) / total_timesteps
        
        self.epsilon = 1e-6  # Small constant for numerical stability
        self.data = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.n_step_buffer = []
        self.index = 0
        self.steps = 0  # Track total steps for beta annealing
        
    def _get_n_step_info(self):
        """Computes N-step return information from recent transitions.
        
        Implements a more robust n-step return calculation by explicitly
        computing discounted rewards and handling terminal states.
        
        Returns:
            tuple: (n_step_reward, final_next_state, final_done)
        """
        # Get the final transition info
        _, _, _, final_next_state, final_done = self.n_step_buffer[-1]
        
        # Calculate n-step discounted reward
        n_step_reward = 0
        for idx, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            # If we hit a terminal state, don't include future rewards
            if idx > 0 and self.n_step_buffer[idx-1][4]:
                break
            n_step_reward += (self.gamma ** idx) * reward
            
        return n_step_reward, final_next_state, final_done
        
    def append(self, state, action, reward, next_state, done):
        """Adds a transition and computes n-step returns.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Wait until we have enough transitions for n-step
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # Compute n-step return
        n_reward, n_next_state, n_done = self._get_n_step_info()
        init_state, init_action = self.n_step_buffer[0][:2]
        
        # Get max priority for new transition
        max_priority = max(np.max(self.priorities), 1.0) if self.data else 1.0
        
        # Store transition
        if len(self.data) < self.capacity:
            self.data.append(None)
            
        self.data[self.index] = (init_state, init_action, n_reward, n_next_state, n_done)
        self.priorities[self.index] = max_priority
        self.index = (self.index + 1) % self.capacity
        
        # Remove oldest transition
        self.n_step_buffer.pop(0)
        
        # Handle episode termination
        if done:
            while len(self.n_step_buffer) > 0:
                n_reward, n_next_state, n_done = self._get_n_step_info()
                init_state, init_action = self.n_step_buffer[0][:2]
                
                if len(self.data) < self.capacity:
                    self.data.append(None)
                    
                self.data[self.index] = (init_state, init_action, n_reward, n_next_state, n_done)
                self.priorities[self.index] = max_priority
                self.index = (self.index + 1) % self.capacity
                self.n_step_buffer.pop(0)

    def sample(self, batch_size):
        N = len(self.data)
        if N == 0:
            raise ValueError("Cannot sample from an empty buffer")
            
        if N == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[:N]
            
        # Convert to sampling probabilities
        probs = (probs + self.epsilon) ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices
        indices = np.random.choice(N, batch_size, p=probs)
        
        # Compute importance sampling weights with current beta
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        
        # Prepare batch
        batch = [self.data[idx] for idx in indices]
        batch = list(map(lambda x: torch.tensor(np.array(x)).to(self.device), zip(*batch)))
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Update beta
        self.steps += 1
        self.beta = min(self.beta_end, 
                       self.beta_start + self.steps * self.beta_increment)
        
        return batch, indices, weights
        
    def update_priorities(self, indices, td_errors):
        """Updates transition priorities based on TD-errors.
        
        Args:
            indices: Indices of transitions to update
            td_errors: New TD-errors for priority computation
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.epsilon
            
    def __len__(self):
        """Returns current buffer size."""
        return len(self.data)
    
    def save_buffer(self, path):
        """Saves buffer state to disk."""
        buffer_data = {
            'capacity': self.capacity,
            'data': self.data,
            'index': self.index,
            'priorities': self.priorities,
            'beta': self.beta
        }
        torch.save(buffer_data, path)
        
    def load_buffer(self, path):
        """Loads buffer state from disk."""
        buffer_data = torch.load(path)
        self.capacity = buffer_data['capacity']
        self.data = buffer_data['data']
        self.index = buffer_data['index']
        self.priorities = buffer_data['priorities']
        self.beta = buffer_data['beta']

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5, factorized=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorized = factorized
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
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
        std = math.sqrt(3 / self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.bias_mu.data.uniform_(-std, std)
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
                weight = self.weight_mu + self.weight_sigma * torch.ger(
                    self.weight_epsilon_output, self.weight_epsilon_input)
            else:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class NoisyHIVTreatmentDQN(nn.Module):
    def __init__(self):
        super(NoisyHIVTreatmentDQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            NoisyLinear(64, 64, std_init=0.5),
            nn.ReLU(),
            NoisyLinear(64, 4, std_init=0.5)
        )
        
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        if not isinstance(x, torch.FloatTensor) and not isinstance(x, torch.cuda.FloatTensor):
            x = torch.FloatTensor(x)
        return self.layers(x)
        
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class ProjectAgent:
    def __init__(self, N_explo=50, T_explo=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Exploration parameters
        self.N_explo = N_explo
        self.T_explo = T_explo
        
        # Network architecture
        self.model = NoisyHIVTreatmentDQN().to(self.device)
        self.target_model = deepcopy(self.model)
        self.training = False
        
        # Configuration
        self.config = {
            'nb_actions': 4,
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'max_episodes': 500,
            'buffer_size': 100000,
            'batch_size': 256,
            'gradient_steps': 2,
            'update_target_strategy': 'ema',
            'update_target_freq': 200,
            'update_target_tau': 0.005,
            'use_Huber_loss': True,
            'n_step': 3  # Added n-step parameter
        }
        
        # Initialize prioritized n-step buffer
        self.memory = NStepPrioritizedBuffer(
            self.config['buffer_size'], 
            self.device,
            n_step=self.config['n_step'],
            gamma=self.config['gamma'],
            total_timesteps=200*self.config["max_episodes"]
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.SmoothL1Loss() if self.config['use_Huber_loss'] else nn.MSELoss()
        self.total_steps = 0
        self.reward_scale = 1e-6
        
    def normalize_state(self, state):
        norm_factors = np.array([1e6, 5e4, 3200.0, 1e2, 2.5e5, 3.532e5])
        epsilon = 1e-8
        normalized = np.log1p(state + epsilon) / np.log1p(norm_factors + epsilon)
        return torch.tensor(normalized, dtype=torch.float32)
        
    def get_sigma_magnitude(self):
        total_sigma = 0
        count = 0
        for module in self.model.modules():
            if isinstance(module, NoisyLinear):
                total_sigma += module.weight_sigma.data.abs().mean().item()
                total_sigma += module.bias_sigma.data.abs().mean().item()
                count += 2
        return total_sigma / count if count > 0 else 0
    
    def get_network_stats(self):
        stats = {
            'weight_mu': [],
            'bias_mu': [],
            'weight_sigma': [],
            'bias_sigma': []
        }
        
        for module in self.model.modules():
            if isinstance(module, NoisyLinear):
                stats['weight_mu'].append(module.weight_mu.data.abs().mean().item())
                stats['bias_mu'].append(module.bias_mu.data.abs().mean().item())
                stats['weight_sigma'].append(module.weight_sigma.data.abs().mean().item())
                stats['bias_sigma'].append(module.bias_sigma.data.abs().mean().item())
        
        return {k: sum(v)/len(v) if v else 0 for k, v in stats.items()}
    
    def get_average_q_values(self, state):
        with torch.no_grad():
            state = self.normalize_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
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
            
            # Sample batch with priorities
            batch, indices, weights = self.memory.sample(self.config['batch_size'])
            states, actions, rewards, next_states, dones = batch
            dones = dones.float()
            # Normalize states
            states = self.normalize_state(states.cpu().numpy())
            next_states = self.normalize_state(next_states.cpu().numpy())
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            
            # Double Q-learning with n-step returns
            with torch.no_grad():
                next_actions = self.model(next_states).argmax(1)
                next_q_values = self.target_model(next_states)
                next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                # n-step returns already computed in buffer
                targets = rewards + (1 - dones) * next_q_values
            
            # Current Q-values
            current_q = self.model(states).gather(1, actions.long().unsqueeze(1))
            
            # Compute weighted loss and TD errors
            td_errors = (targets.unsqueeze(1) - current_q).detach().cpu().numpy()
            loss = (weights * F.smooth_l1_loss(current_q, targets.unsqueeze(1), reduction='none')).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update priorities
            self.memory.update_priorities(indices, td_errors)

    def act(self, state, use_random=False):
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
        if self.config['update_target_strategy'] == 'replace':
            if self.total_steps % self.config['update_target_freq'] == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        else:  # EMA update
            tau = self.config['update_target_tau']
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train(self, env):
        """Train the agent with initial exploration phase"""
        episode_returns = []
        cumulative_reward = 0
        state, _ = env.reset()
        action_counts = np.zeros(4)
        
        # Exploration phase tracking
        steps_since_policy_change = 0
        current_exploration_policy = np.random.randint(0, 4)  # Initial random policy

        with tqdm(total=self.config["max_episodes"], desc="Training Progress", unit="episode") as pbar:
            while len(episode_returns) < self.config["max_episodes"]:
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
                
                # Training steps
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
                        f"β: {self.memory.beta:.3f}, "  # Added beta tracking
                        f"Phase: {'Exploration' if in_exploration else 'Training'}")

                    cumulative_reward = 0
                    action_counts = np.zeros(4)
                    state, _ = env.reset()

                    # Save checkpoint every 500 episodes
                    if len(episode_returns) % 500 == 0:
                        checkpoint_path = f"saved_model_{len(episode_returns)}.pt"
                        self.save(checkpoint_path)
                        print(f"Saved checkpoint at episode {len(episode_returns)}")

                else:
                    state = next_state

        return episode_returns
            
    def save(self, path):
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
                print(f"Model and replay buffer loaded successfully")
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
    
    save_path = "saved_model.pt"

    # Train the agent
    print("Training the agent...")
    episode_returns = agent.train(env)

    # Save the trained model
    print(f"Saving the trained model to {save_path}...")
    agent.save(save_path)

    print("Training complete!")