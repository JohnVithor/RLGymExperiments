from collections import defaultdict
import numpy as np

class SarsaAgent:
    def __init__(
        self,
        action_space,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ) -> None:
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            action_space: The action space of the env that the agent will act upon
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.action_space = action_space
        self.q_values = defaultdict(lambda: np.zeros(action_space.n))
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = initial_epsilon
        self.epsilon_decay: float = epsilon_decay
        self.final_epsilon: float = final_epsilon
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
        future_q_value: float = (not terminated) * self.q_values[next_obs][next_action]

        temporal_difference: float = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        self.q_values[obs][action] = self.q_values[obs][action] + self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)