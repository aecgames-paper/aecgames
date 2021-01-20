
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
from cyclic_reward_wrapper import cyclically_expansive_learning

# for APEX-DQN
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import ray.rllib.agents.ppo as ppo

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


def make_env_creator(env_name, game_env):
    def env_creator(args):
        env = game_env.env()
        if env_name == 'curriculum':
            env = cyclically_expansive_learning(env, [(0,1*8), (10000000,2*8), (30000000,3*8), (50000000,8*8)])
        env = flatten_v0(env)
        return env
    return env_creator


def get_env(env_name):
    if env_name == 'unpruned':
        game_env = unpruned_pursuit_v0
    elif env_name == 'pruned':
        game_env = pursuit_v3
    elif env_name == 'curriculum':
        game_env = pursuit_v3
    else:
        raise TypeError("{} environment not supported!".format(game_env))
    return game_env

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    assert len(sys.argv) == 4, "Input the learning method as the second argument"
    env_name = sys.argv[1]
    method = sys.argv[2]
    checkpoint_path = sys.argv[3]

    game_env = get_env(env_name) 
    env_creator = make_env_creator(env_name, game_env)

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
        #"vf_share_layers": True,

        # Method specific

        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (
                lambda agent_id: policy_ids[0]),
        },
    }


    #ray.init()
    #trainer = ppo.PPOTrainer(config=config, env=env_name)
    #trainer.restore("/home/samaisantos/results_unpruned/pruned/PPO/PPO_pruned_5ed91_00000_0_2021-01-03_23-42-41/checkpoint_27700/checkpoint-27700")

    tune.run(
        'PPO',
        name="PPO",
        restore='/home/samaisantos/results_unpruned/'+env_name+'/PPO/'+checkpoint_path+'/checkpoint_27700/checkpoint-27700',
        stop={"episodes_total": 60000},
        checkpoint_freq=100,
        local_dir="~/results_unpruned_restore/"+env_name,
        config=config,
    )



'''
    if method == "ADQN":
        # APEX-DQN
        tune.run(
            "APEX",
            name="ADQN",
            stop={"episodes_total": 60000},
            checkpoint_freq=100,
            local_dir="~/results_unpruned/"+env_name,
            config={

                ### PREVIOUS PARAMS
                # Enviroment specific
                #"env": env_name,

                # General
                #"log_level": "INFO",
                #"num_gpus": 1,
                #"num_workers": 8,
                #"num_envs_per_worker": 8,
                #"learning_starts": 1000,
                #"buffer_size": int(1e5),
                #"compress_observations": True,
                #"rollout_fragment_length": 20,
                #"train_batch_size": 512,
                #"gamma": .99,

                # Method specific
                

                ### NEW PARAMS
                # Enviroment specific
                "env": env_name,
                "double_q": True,
                "dueling": True,
                "num_atoms": 1,
                "noisy": False,
                "n_step": 3,
                "lr": 0.0001,
                #"lr": 0.0000625,
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
            local_dir="~/results_unpruned_restore/"+env_name,
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
                #"vf_share_layers": True,

                # Method specific

                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    # psuedo-rainbow DQN
    elif method == "RDQN":
        tune.run(
            "DQN",
            name="RDQN",
            stop={"episodes_total": 60000},
            checkpoint_freq=100,
            local_dir="~/results_unpruned/"+env_name,
            config={

                # Enviroment specific
                "env": "pursuit",

                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,

                # Method specific
                "num_atoms": 51,
                "dueling": True,
                "double_q": True,
                "n_step": 2,
                "batch_mode": "complete_episodes",
                "prioritized_replay": True,

                # # alternative 1
                # "noisy": True,
                # alternative 2
                "parameter_noise": True,

                # based on expected return
                "v_min": 0,
                "v_max": 1500,

                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

'''
