from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract base class for all agents.
    """

    @abstractmethod
    def train_step(self, obs, action, reward, terminal):
        """
        Perform a training step.
        """
        pass
        
    @abstractmethod
    def save(self, path, logs=None):
        """
        Save the agent's state.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load the agent's state.
        """
        pass

    @abstractmethod
    def policy(self, obs, state, last_action):
        """
        Compute action and next state given observation and previous state.
        Returns: action (numpy), next_state (tuple/tensor), env_action
        """
        pass
