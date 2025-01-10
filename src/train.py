import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
import itertools
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

class PredictionNetwork(nn.Module):
    def __init__(self, in_dim: int, nf: int, out_dim: int):
        """Initialization."""
        super(PredictionNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, nf),
            nn.LeakyReLU(),
            nn.Linear(nf, nf),
            nn.LeakyReLU(),
            nn.Linear(nf, nf),
            nn.LeakyReLU(),
            nn.Linear(nf, out_dim)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)

class ReplayBuffer:
    """Memory buffer for storing and sampling training experiences with n-step returns."""
    
    def __init__(
        self, 
        state_dimension: int,
        capacity: int,
        sample_size: int = 32,
        steps_per_return: int = 3,
        discount: float = 0.99
    ):
        # Main storage arrays
        self.states = np.zeros((capacity, state_dimension), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dimension), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)
        
        # Configuration
        self.capacity = capacity
        self.sample_size = sample_size
        self.discount = discount
        self.steps_per_return = steps_per_return
        
        # State tracking
        self.write_pos = 0
        self.stored_experiences = 0
        
        # N-step return tracking
        self.step_history = {
            'states': np.zeros((steps_per_return, state_dimension), dtype=np.float32),
            'actions': np.zeros(steps_per_return, dtype=np.float32),
            'rewards': np.zeros(steps_per_return, dtype=np.float32)
        }
        self.history_index = 0
        self.history_complete = False
        
    def clear_history(self):
        """Reset the n-step history tracking."""
        for key in self.step_history:
            self.step_history[key].fill(0)
        self.history_index = 0
        self.history_complete = False
        
    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ) -> bool:
        """
        Store a new experience in the buffer and handle n-step returns.
        Returns True if an experience was actually stored (after n-steps are collected).
        """
        # Update step history
        self.step_history['states'][self.history_index] = state
        self.step_history['actions'][self.history_index] = action
        self.step_history['rewards'][self.history_index] = reward
        
        self.history_index = (self.history_index + 1) % self.steps_per_return
        self.history_complete = self.history_complete or (self.history_index == 0)
        
        # Process n-step return when history is complete
        if self.history_complete:
            # Calculate n-step discounted return
            n_step_return = 0
            discount_factor = 1
            
            # Collect rewards in correct order based on circular buffer
            reward_sequence = itertools.chain(
                self.step_history['rewards'][self.history_index:],
                self.step_history['rewards'][:self.history_index]
            )
            
            # Calculate discounted return
            for step_reward in reward_sequence:
                n_step_return += discount_factor * step_reward
                discount_factor *= self.discount
            
            # Store the experience with n-step return
            hist_idx = self.history_index
            self.states[self.write_pos] = self.step_history['states'][hist_idx]
            self.next_states[self.write_pos] = next_state
            self.actions[self.write_pos] = self.step_history['actions'][hist_idx]
            self.rewards[self.write_pos] = n_step_return
            self.terminals[self.write_pos] = terminal
            
            # Update buffer state
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.stored_experiences = min(self.stored_experiences + 1, self.capacity)
        
        # Reset history if episode ends
        if terminal:
            self.clear_history()
            
        return self.history_complete
    
    def get_batch(self) -> dict:
        """Sample a random batch of experiences from the buffer."""
        indices = np.random.choice(
            self.stored_experiences,
            size=self.sample_size,
            replace=False
        )
        
        return {
            'states': self.states[indices],
            'next_states': self.next_states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'terminals': self.terminals[indices]
        }
    
    def current_size(self) -> int:
        """Return the current number of experiences stored."""
        return self.stored_experiences

class ProjectAgent:
    """Agent for the HIV treatment environment using DQN."""
    
    def __init__(self):
        # Network parameters
        self.obs_dim = 6
        self.action_dim = 4
        self.hidden_dim = 1024
        self.stack_n_prev_frames = 3  # Number of previous frames to stack
        
        # Training parameters
        self.batch_size = 2048
        self.learning_rate = 2e-4
        self.memory_size = int(1e6)
        self.gamma = 0.99
        self.tau = 0.005
        self.n_step_return = 1
        self.grad_clip = 1000.0
        self.target_update_freq = 1000
        
        # Exploration parameters
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay = 1 / 300
        self.epsilon = self.max_epsilon
        
        # Frame stacking
        self.augmented_obs_dim = self.obs_dim * (self.stack_n_prev_frames + 1)
        self.augmented_obs = None
        self.t = 0  # Time step counter for episodes
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks with augmented observation dimension
        self.dqn = PredictionNetwork(
            in_dim=self.augmented_obs_dim,  # Increased input dimension for stacked frames
            nf=self.hidden_dim,
            out_dim=self.action_dim
        ).to(self.device)
        
        self.dqn_target = PredictionNetwork(
            in_dim=self.augmented_obs_dim,
            nf=self.hidden_dim,
            out_dim=self.action_dim
        ).to(self.device)

        self.dqn_target.load_state_dict(self.dqn.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer with augmented observation dimension
        self.memory = ReplayBuffer(
            self.augmented_obs_dim,
            self.memory_size,
            self.batch_size,
            self.n_step_return,
            self.gamma
        )
        
        # Training tracking
        self.total_steps = 0
        self.is_training = False

    def preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        """Preprocess the observation by reordering and applying log10."""
        # Reorder the observation: [T1, T1*, T2, T2*, V, E] -> [T1, T2, T1*, T2*, V, E]
        reorder_idx = np.array([0, 2, 1, 3, 4, 5])
        return np.log10(observation[reorder_idx])

    def update_stacked_frames(self, observation: np.ndarray) -> np.ndarray:
        """Update the stacked frames with new observation."""
        if self.t == 0:
            # At the start of an episode, stack the same observation multiple times
            self.augmented_obs = np.stack([observation] * (self.stack_n_prev_frames + 1), axis=0)
        else:
            # During episode, shift the frames and add new observation
            self.augmented_obs[:self.stack_n_prev_frames, :] = self.augmented_obs[1:]
            self.augmented_obs[-1, :] = observation
        
        self.t = (self.t + 1) % 200  # Reset counter at end of episode
        return self.augmented_obs.reshape(-1)  # Flatten the stacked frames

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        """Select an action from the input state."""
        # Preprocess the observation
        processed_obs = self.preprocess_observation(observation)
        
        # Update stacked frames
        stacked_obs = self.update_stacked_frames(processed_obs)
        
        # Exploration during training
        if use_random and np.random.rand() < self.epsilon:
            network_action = np.random.randint(self.action_dim)
        else:
            # Convert to tensor and get network prediction
            state = torch.FloatTensor(stacked_obs).to(self.device)
            
            self.dqn.eval()
            with torch.no_grad():
                q_values = self.dqn(state)
                network_action = q_values.argmax().item()
            self.dqn.train(self.is_training)
        
        # Map network action to environment action
        action_map = np.array([0, 2, 1, 3])  # Maps network actions to environment actions
        return action_map[network_action]

    def train(self,env, num_episodes: int = 1000) -> None:
        """Train the agent."""
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            self.t = 0  # Reset frame counter at start of episode
            episode_reward = 0
            
            while True:
                action = self.act(state, use_random=True)
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Get processed and stacked observations
                processed_state = self.preprocess_observation(state)
                processed_next_state = self.preprocess_observation(next_state)
                stacked_state = self.augmented_obs.reshape(-1)
                
                # Update stacked frames for next state
                next_stacked_state = self.update_stacked_frames(processed_next_state)
                
                # Store transition in memory
                self.memory.store(stacked_state, action, reward, next_stacked_state, done)
                
                # Train if enough samples
                if len(self.memory) > self.batch_size:
                    self._train_step()
                
                state = next_state
                episode_reward += reward
                
                if done or truncated:
                    break
                    
            # Update epsilon
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay
            )
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.2f}")

    def _train_step(self) -> None:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()
        
        # Convert to tensor
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        
        # Get current Q value
        current_q_value = self.dqn(state).gather(1, action)
        
        # Get next Q value with Double DQN
        next_q_value = self.dqn_target(next_state).gather(
            1, self.dqn(next_state).argmax(dim=1, keepdim=True)
        ).detach()
        
        # Compute target Q value
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_value, target)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip)
        self.optimizer.step()
        
        self.total_steps += 1
        
        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())

    def save(self, path: str) -> None:
        """Save the model to the specified path."""
        torch.save({
            'dqn': self.dqn.state_dict(),
            'dqn_target': self.dqn_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }, path)

    def load(self) -> None:
        """Load the model from a hardcoded path."""
        # Load the full checkpoint
        saved_model = torch.load('saved_model_best_1250_light.pt', map_location=self.device)
        # Extract and load just the DQN state dict
        if isinstance(saved_model, dict) and 'dqn' in saved_model:
            # If checkpoint is in the format saved during training
            self.dqn.load_state_dict(saved_model['dqn'])
        else:
            # If checkpoint is just the state dict
            self.dqn.load_state_dict(saved_model)
        
        self.is_training = False
        self.dqn.eval()  # Set to evaluation mode
        self.t = 0  # Reset frame counter

if __name__ == "__main__":
    # Example usage
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
    agent.is_training = True
    # Training configuration
    max_episodes = 1500
    save_path = "saved_model3.pt"

    # Train the agent
    print("Training the agent...")
    agent.train(env, max_episodes)

    # Save the trained model
    print(f"Saving the trained model to {save_path}...")
    agent.save(save_path)

    print("Training complete!")