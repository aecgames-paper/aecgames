
import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

import sys
import gym
import random
import numpy as np

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.rllib.env import PettingZooEnv
from pettingzoo.sisl import unpruned_pursuit_v0, pursuit_v3

from supersuit import flatten_v0

# for APEX-DQN
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

tf1, tf, tfv = try_import_tf()

class MLPModelV2(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="my_model"):
        super(MLPModelV2, self).__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # Simplified to one layer.
        input_layer = tf.keras.layers.Input(
                obs_space.shape,
                dtype=obs_space.dtype)
        layer_1 = tf.keras.layers.Dense(
                400,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(input_layer)
        layer_2 = tf.keras.layers.Dense(
                300,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(layer_1)
        output = tf.keras.layers.Dense(
                num_outputs,
                activation=None,
                kernel_initializer=normc_initializer(0.01))(layer_2)
        value_out = tf.keras.layers.Dense(
                1,
                activation=None,
                name="value_out",
                kernel_initializer=normc_initializer(0.01))(layer_2)
        self.base_model = tf.keras.Model(input_layer, [output, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return  tf.reshape(self._value_out, [-1])


def make_env_creator(game_env):
    def env_creator(args):
        env = game_env.env()
        env = flatten_v0(env)
        return env
    return env_creator


def get_env(env_name):
    if env_name == 'unpruned':
        game_env = unpruned_pursuit_v0
    elif env_name == 'pruned':
        game_env = pursuit_v3
    else:
        raise TypeError("{} environment not supported!".format(game_env))
    return game_env

if __name__ == "__main__":
    # PPO  - PPO
    # ADQN - Apex DQN

    assert len(sys.argv) == 3, "Input the learning method as the second argument"
    env_name = sys.argv[1]
    method = sys.argv[2]

    game_env = get_env(env_name) 
    env_creator = make_env_creator(game_env)

    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    test_env = PettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "MLPModelV2",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)
    policies = {"policy_0": gen_policy(0)}

    # for all methods
    policy_ids = list(policies.keys())

    if method == "ADQN":
        # APEX-DQN
        tune.run(
            "APEX",
            name="ADQN",
            stop={"episodes_total": 60000},
            checkpoint_freq=100,
            local_dir="~/results_unpruned/"+env_name,
            config={
                # Enviroment specific
                "env": env_name,
                "double_q": True,
                "dueling": True,
                "num_atoms": 1,
                "noisy": False,
                "n_step": 3,
                "lr": 0.0001,
                "adam_epsilon": 1.5e-4,
                "buffer_size": int(1e5),
                "exploration_config": {
                    "final_epsilon": 0.01,
                    "epsilon_timesteps": 200000,
                },
                "prioritized_replay": True,
                "prioritized_replay_alpha": 0.5,
                "prioritized_replay_beta": 0.4,
                "final_prioritized_replay_beta": 1.0,
                "prioritized_replay_beta_annealing_timesteps": 2000000,

                "num_gpus": 1,

                "log_level": "ERROR",
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "rollout_fragment_length": 32,
                "train_batch_size": 512,
                "target_network_update_freq": 50000,
                "timesteps_per_iteration": 25000,
                "learning_starts": 80000,
                "compress_observations": False,
                "gamma": 0.99,
 
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "PPO":
        tune.run(
            "PPO",
            name="PPO",
            stop={"episodes_total": 60000},
            checkpoint_freq=100,
            local_dir="~/results_unpruned/"+env_name,
            config={
                # Enviroment specific
                "env": env_name,
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "compress_observations": False,
                "gamma": .99,

                "lambda": 0.95,
                "kl_coeff": 0.5,
                "clip_rewards": True,
                "clip_param": 0.1,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
                "train_batch_size": 5000,
                "rollout_fragment_length": 100,
                "sgd_minibatch_size": 500,
                "num_sgd_iter": 10,
                "batch_mode": 'truncate_episodes',

                # Method specific
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

