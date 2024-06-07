from dm_control import suite
from dm_env import specs
import numpy as np


class DMControlWrapper:
    def __init__(self, domain_name, task_name):
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()

    def reset(self):
        time_step = self.env.reset()
        state = self._flatten_observation(time_step.observation)
        return state

    def get_random_action(self):
        action = np.random.uniform(self.action_spec.minimum, self.action_spec.maximum, self.action_spec.shape)
        return action

    def step(self, action):
        time_step = self.env.step(action)
        next_state = self._flatten_observation(time_step.observation)
        reward = time_step.reward
        terminal = time_step.last()
        return next_state, reward, terminal

    def set_random_seed(self, seed):
        np.random.seed(seed)

    def render(self):
        frame = self.env.physics.render(camera_id=0, height=240, width=320)
        return frame

    def get_state_dims(self):
        dims = sum(np.prod(spec.shape) for spec in self.observation_spec.values())
        return (dims,)

    def get_state_bounds(self):
        lower_bounds = []
        upper_bounds = []
        for spec in self.observation_spec.values():
            lower_bounds.append(np.tile(spec.minimum, np.prod(spec.shape)))
            upper_bounds.append(np.tile(spec.maximum, np.prod(spec.shape)))
        return np.concatenate(lower_bounds), np.concatenate(upper_bounds)

    def get_action_dims(self):
        return self.action_spec.shape

    def get_action_bounds(self):
        return self.action_spec.minimum, self.action_spec.maximum

    def close(self):
        pass  # dm_control environments don't need explicit closing

    def _flatten_observation(self, observation):
        """Flattens the observation dictionary into a single numpy array."""
        return np.concatenate([np.ravel(ob) for ob in observation.values()])


# Example wrappers for specific dm_control environments
class CartpoleSwingupWrapper(DMControlWrapper):
    def __init__(self):
        super().__init__('cartpole', 'swingup')

    def normalise_state(self, state):
        # Custom normalization for CartpoleSwingup environment if needed
        return state

    def normalise_reward(self, reward):
        # Custom normalization for CartpoleSwingup environment if needed
        return reward / 10.0


class WalkerWalkWrapper(DMControlWrapper):
    def __init__(self):
        super().__init__('walker', 'walk')

    def normalise_state(self, state):
        # Custom normalization for WalkerWalk environment if needed
        return state

    def normalise_reward(self, reward):
        # Custom normalization for WalkerWalk environment if needed
        return reward / 10.0
