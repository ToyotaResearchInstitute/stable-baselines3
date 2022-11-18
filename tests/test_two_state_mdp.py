from cgi import test
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.envs import TwoStateMDP, RolloutInfoWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

DIM = 1


def rollout_buffer_wrapper_factory():
    """Returns a function that wraps the environment in a RolloutInfoBuffer with custom logger"""

    def f(env: gym.Env) -> gym.Env:
        return RolloutInfoWrapper(env)

    return f


# @pytest.mark.parametrize("model_class", [PPO])
# @pytest.mark.parametrize("env", [TwoStateMDP])
def test_discrete():
    # when reward_coeff is positive, then reward is positive when action == 0
    # when reward_coeff is negative, then reward is positive when action == 1
    # env_ = TwoStateMDP(reward_coeff=1, ep_length=100)
    env = TwoStateMDP(reward_coeff=1, ep_length=100)
    env = Monitor(env)
    post_wrappers = [rollout_buffer_wrapper_factory()]
    for wrapper in post_wrappers:
        env = wrapper(env)
    env_ = VecNormalize(DummyVecEnv([lambda: env]))

    policy_kwargs = {"use_sde": False}
    batch_size = 2048
    rl_kwargs = dict(
        # For recommended PPO hyperparams in each environment, see:
        # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
        learning_rate=1e-2,
        # N_timesteps=1e7,
        batch_size=1024,
        n_epochs=500,
        ent_coef=0.01,
        n_steps=2048,
        verbose=1,
        tensorboard_log="output",
        device="cpu",
        seed=0,
    )
    model = PPO("MlpPolicy", env_, policy_kwargs=policy_kwargs, **rl_kwargs).learn(total_timesteps=500000)

    evaluate_policy(model, env_, n_eval_episodes=20, reward_threshold=90, warn=False)
    # obs = env.reset()

    # assert np.shape(model.predict(obs)[0]) == np.shape(obs)


if __name__ == "__main__":  # pragma: no cover
    test_discrete()
