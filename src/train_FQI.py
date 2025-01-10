import numpy as np
from sklearn.ensemble import RandomForestRegressor
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import tqdm
import os
import joblib

class ProjectAgent:
    def __init__(self):
        # FQI parameters
        self.n_estimators = 100  # Number of trees in random forest
        self.n_iterations = 50   # Number of FQI iterations
        self.gamma = 0.98       # Discount factor
        self.Q = None          # Current Q-function approximator
        
        # Storage paths
        self.samples_path = "stored_samples.joblib"
        self.model_path = "saved_model.joblib"

    def collect_samples(self, n_samples=10000, force_recollect=False):
        """Collect samples using random policy or load existing ones"""
        if os.path.exists(self.samples_path) and not force_recollect:
            print("Loading existing samples...")
            return joblib.load(self.samples_path)
        
        print("Collecting new samples...")
        env = TimeLimit(HIVPatient(), max_episode_steps=200)
        samples = []
        state, _ = env.reset()
        
        pbar = tqdm(total=n_samples, desc="Collecting samples")
        while len(samples) < n_samples:
            action = np.random.randint(4)  # Random action
            next_state, reward, done, truncated, _ = env.step(action)
            samples.append((state, action, reward, next_state, done))
            pbar.update(1)
            
            if done or truncated:
                state, _ = env.reset()
            else:
                state = next_state
        
        pbar.close()
        
        # Store samples for future use
        print("Storing samples for future use...")
        joblib.dump(samples, self.samples_path)
                
        return samples

    def train(self):
        # Get or collect samples
        samples = self.collect_samples()
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])

        # Prepare X (state-action pairs) for random forest
        X = np.hstack([states, actions.reshape(-1, 1)])

        # Initial Q-function (zero everywhere)
        print("Initializing Q-function...")
        self.Q = RandomForestRegressor(n_estimators=self.n_estimators)
        self.Q.fit(X, rewards)  # Initialize with just rewards

        # FQI iterations
        print("Starting FQI iterations...")
        for iter in tqdm(range(self.n_iterations), desc="FQI Progress"):
            # Compute next state maximum values
            next_values = np.zeros(len(samples))
            for a in range(4):
                next_X = np.hstack([next_states, np.full((len(samples), 1), a)])
                next_values = np.maximum(next_values, self.Q.predict(next_X))

            # Compute targets
            targets = rewards + self.gamma * (1 - dones) * next_values

            # Fit new Q-function
            new_Q = RandomForestRegressor(n_estimators=self.n_estimators)
            new_Q.fit(X, targets)
            self.Q = new_Q

            # Log some statistics every 5 iterations
            if (iter + 1) % 5 == 0:
                avg_target = np.mean(targets)
                max_target = np.max(targets)
                print(f"\nIteration {iter + 1} stats:")
                print(f"Average target value: {avg_target:.2f}")
                print(f"Max target value: {max_target:.2f}")

    def act(self, observation, use_random=False):
        """Choose action according to current Q-function"""
        if use_random:
            return np.random.randint(4)
        print("Predicting")
        values = np.array([
            self.Q.predict(np.hstack([observation, [a]]).reshape(1, -1))[0]
            for a in range(4)
        ])
        return np.argmax(values)

    def save(self, path):
        """Save the agent's Q-function"""
        joblib.dump(self.Q, self.model_path)

    def load(self):
        """Load the agent's Q-function"""
        try:
            print("Loading model...")
            self.Q = joblib.load(self.model_path)
            print("Model loaded.")
        except:
            print("No saved model found.")

if __name__ == "__main__":
    # Create and train agent
    agent = ProjectAgent()
    agent.train()
    
    # Save the trained agent
    agent.save("saved_model_FQI.joblib")