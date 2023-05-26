from collections import defaultdict
import numpy as np

class QLearningLambdaAgent:
    def __init__(
        self,
        action_space,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        lambda_factor: float,
        discount_factor: float = 0.95,
    ) -> None:
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            action_space: The environment's action space that the agent will act upon
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            lambda_factor: The eligibility trace value
            discount_factor: The discount factor for computing the Q-value
        """
        self.action_space = action_space
        self.q_values = defaultdict(lambda: np.zeros(action_space.n))
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay: float = epsilon_decay
        self.final_epsilon: float = final_epsilon
        self.lambda_factor: float = lambda_factor
        self.e_values = defaultdict(lambda: np.zeros(action_space.n)) # Eligibility trace
        self.training_error: list = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_values[obs])

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
        next_action: int
    ) -> None:
        """Updates the Q-value of an action."""
        best_next_action: int = np.argmax(self.q_values[next_obs])
        future_q_value: float = (not terminated) * self.q_values[next_obs][best_next_action]
        temporal_difference: float = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        
        # Update Eligibility trace
        self.e_values[obs][action] += 1

        for s in self.q_values:
            for a in range(self.action_space.n):
                # Update Q value based on eligibility trace
                self.q_values[s][a] += self.learning_rate * temporal_difference * self.e_values[s][a]
                # Decay eligibility trace if best action is taken
                self.e_values[s][a] *= (next_action == best_next_action) * self.discount_factor * self.lambda_factor
        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        self.epsilon: float = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
