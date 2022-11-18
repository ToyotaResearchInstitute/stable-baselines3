from typing import Optional, Union, List

import numpy as np
from gym import Env, Space, Wrapper
from gym.utils import seeding
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from stable_baselines3.common.logger import Logger


class RolloutInfoWrapper(Wrapper):
    """Add the entire episode's rewards and observations to `info` at episode end.

    Whenever done=True, `info["rollouts"]` is a dict with keys "obs" and "rews", whose
    corresponding values hold the NumPy arrays containing the raw observations and
    rewards seen during this episode.
    """

    def __init__(self, env: Env, custom_logger: Logger = None):
        """Builds RolloutInfoWrapper.

        Args:
            env: Environment to wrap.
        """
        super().__init__(env)
        self._obs = None
        self._rews = None
        self._rew_components = None
        self._custom_logger = custom_logger

    def reset(self, **kwargs):
        new_obs = super().reset()
        self._obs = [new_obs]
        self._rews = []
        self._rew_components = []
        return new_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        # print("Reward in rollout wrapper", rew, info)
        self._obs.append(obs)
        self._rews.append(rew)
        if "reward_components" in info:
            self._rew_components.append(info)  # list of dict

        if done:
            assert "rollout" not in info
            if len(self._rew_components) > 0:
                info["rollout"] = {"obs": np.stack(self._obs), "rews": np.stack(self._rews)}

                # sum of reward components for entire episode.
                for k in self._rew_components[0]["reward_components"].keys():
                    info["rollout"][k] = np.sum([rc["reward_components"][k] for rc in self._rew_components])
            else:
                info["rollout"] = {"obs": np.stack(self._obs), "rews": np.stack(self._rews)}
                print(info)
            if len(self._rew_components) > 0 and self._custom_logger is not None:
                with self._custom_logger.accumulate_means("avg_episodes"):
                    self._custom_logger.record(
                        "rollout/action_reward",
                        np.sum([rc["reward_components"]["action_reward"] for rc in self._rew_components]),
                    )

        return obs, rew, done, info


class TwoStateMDP(Env):
    def __init__(self, reward_coeff=-1, ep_length=100) -> None:
        self.action_space = Discrete(2)
        # self.observation_space = Discrete(2)
        self.observation_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.ep_length = ep_length
        self.current_step = 0
        self.reward_coeff = reward_coeff
        self.reset()

    def reset(self) -> None:
        self.current_step = 0
        self.state = 0
        return self.state

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        full_action = [np.random.choice([0, 1]), action]
        print("")
        print("previous state", self.state)
        self._compute_next_state(full_action)
        reward = self._compute_reward(full_action)
        self.current_step += 1
        print("reward", reward)
        done = self.current_step >= self.ep_length

        return self.state, reward, done, {}

    def _compute_reward(self, full_action) -> float:
        is_action_zero = full_action[-1] == 0
        is_action_non_zero = full_action[-1] == 1
        assert is_action_zero == (not is_action_non_zero)
        # when reward_coeff is positive, then reward is positive when action == 0
        # when reward_coeff is negative, then reward is positive when action == 1
        reward = self.reward_coeff * (int(is_action_zero) - int(is_action_non_zero))
        return reward

    def _compute_next_state(self, full_action):
        distraction_action = full_action[0]
        if distraction_action == 0:
            self.state = 1
        elif distraction_action == 1:
            self.state = 0
        print("distraction_action", full_action[0])
        print("hmi_action", full_action[-1])
        print("state_before alert", self.state)
        hmi_action = full_action[-1]
        if hmi_action == 1:
            self.state = 0
        print("state after alert", self.state)

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        print("print env seed", self.np_random, seed)
        return [seed]


class IdentityEnv(Env):
    def __init__(self, dim: Optional[int] = None, space: Optional[Space] = None, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the action and observation dimension you want
            to learn. Provide at most one of ``dim`` and ``space``. If both are
            None, then initialization proceeds with ``dim=1`` and ``space=None``.
        :param space: the action and observation space. Provide at most one of
            ``dim`` and ``space``.
        :param ep_length: the length of each episode in timesteps
        """
        if space is None:
            if dim is None:
                dim = 1
            space = Discrete(dim)
        else:
            assert dim is None, "arguments for both 'dim' and 'space' provided: at most one allowed"

        self.action_space = self.observation_space = space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self) -> GymObs:
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

    def _get_reward(self, action: Union[int, np.ndarray]) -> float:
        return 1.0 if np.all(self.state == action) else 0.0

    def render(self, mode: str = "human") -> None:
        pass


class IdentityEnvBox(IdentityEnv):
    def __init__(self, low: float = -1.0, high: float = 1.0, eps: float = 0.05, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param low: the lower bound of the box dim
        :param high: the upper bound of the box dim
        :param eps: the epsilon bound for correct value
        :param ep_length: the length of each episode in timesteps
        """
        space = Box(low=low, high=high, shape=(1,), dtype=np.float32)
        super().__init__(ep_length=ep_length, space=space)
        self.eps = eps

    def step(self, action: np.ndarray) -> GymStepReturn:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _get_reward(self, action: np.ndarray) -> float:
        return 1.0 if (self.state - self.eps) <= action <= (self.state + self.eps) else 0.0


class IdentityEnvMultiDiscrete(IdentityEnv):
    def __init__(self, dim: int = 1, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the dimensions you want to learn
        :param ep_length: the length of each episode in timesteps
        """
        space = MultiDiscrete([dim, dim])
        super().__init__(ep_length=ep_length, space=space)


class IdentityEnvMultiBinary(IdentityEnv):
    def __init__(self, dim: int = 1, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the dimensions you want to learn
        :param ep_length: the length of each episode in timesteps
        """
        space = MultiBinary(dim)
        super().__init__(ep_length=ep_length, space=space)


class FakeImageEnv(Env):
    """
    Fake image environment for testing purposes, it mimics Atari games.

    :param action_dim: Number of discrete actions
    :param screen_height: Height of the image
    :param screen_width: Width of the image
    :param n_channels: Number of color channels
    :param discrete: Create discrete action space instead of continuous
    :param channel_first: Put channels on first axis instead of last
    """

    def __init__(
        self,
        action_dim: int = 6,
        screen_height: int = 84,
        screen_width: int = 84,
        n_channels: int = 1,
        discrete: bool = True,
        channel_first: bool = False,
    ):
        self.observation_shape = (screen_height, screen_width, n_channels)
        if channel_first:
            self.observation_shape = (n_channels, screen_height, screen_width)
        self.observation_space = Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        if discrete:
            self.action_space = Discrete(action_dim)
        else:
            self.action_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.ep_length = 10
        self.current_step = 0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        return self.observation_space.sample()

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        reward = 0.0
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.observation_space.sample(), reward, done, {}

    def render(self, mode: str = "human") -> None:
        pass
