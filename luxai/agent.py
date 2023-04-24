"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from lux.config import EnvConfig

from wrappers import IceMinerUnitDiscreteController, RubbleMinerUnitDiscreteController, OreMinerUnitDiscreteController
from wrappers import IceMinerUnitObservationWrapper, RubbleMinerUnitObservationWrapper, OreMinerUnitObservationWrapper
from wrappers import collisionHandler

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH_ICE = "./best_model_ice"
MODEL_WEIGHTS_RELATIVE_PATH_RUBBLE = "./best_model_rubble"
MODEL_WEIGHTS_RELATIVE_PATH_ORE = "./best_model_ore"

class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        # Robot holders
        self.water_units = {}
        self.ore_units = {}
        self.rubble_units = {}

        directory = osp.dirname(__file__)
        self.iceMinerPolicy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH_ICE))
        self.rubbleMinerPolicy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH_RUBBLE))
        self.oreMinerPolicy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH_ICE))

        self.iceMinerController = IceMinerUnitDiscreteController(self.env_cfg)
        self.rubbleMinerController = RubbleMinerUnitDiscreteController(self.env_cfg)
        self.oreMinerController = OreMinerUnitDiscreteController(self.env_cfg)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        if obs["teams"][self.player]["metal"] == 0:
            return dict()
        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False

        ice_diff = np.diff(obs["board"]["ice"])
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]

            area = 3
            for x in range(area):
                for y in range(area):
                    check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        if not done_search:
            pos = spawn_loc

        metal = obs["teams"][self.player]["metal"]
        return dict(spawn=pos, metal=metal, water=metal)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)

        # Get commandable items
        shared_obs = raw_obs[self.player]
        factories = shared_obs["factories"][self.player]
        efactories = shared_obs["factories"][self.opp_player]
        units = shared_obs["units"][self.player]
        eunits = shared_obs["units"][self.opp_player]

        # Create a water guy if not exists for each factory
        if len(units) == 0:
            for unit_id in factories.keys():
                lux_action[unit_id] = 1  # build a single heavy until 2* robots present
                self.water_units[f'unit_{len(units)+len(factories)+len(eunits)+len(efactories)}'] = unit_id

        # Get observation for ice miner
        ice_obs = IceMinerUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        ice_obs = ice_obs[self.player]
        ice_obs = th.from_numpy(ice_obs).float()

        # Get observation for rubble miner
        rubble_obs = RubbleMinerUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        rubble_obs = rubble_obs[self.player]
        rubble_obs = th.from_numpy(rubble_obs).float()

        # Get observation for ore miner
        ore_obs = OreMinerUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        ore_obs = ore_obs[self.player]
        ore_obs = th.from_numpy(ore_obs).float()

        with th.no_grad():

            # to improve performance, we have rule based action mask generators for the controllers used
            # which will force the agents to generate actions that are valid only.
            ice_action_mask = (
                th.from_numpy(self.iceMinerController.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            rubble_action_mask = (
                th.from_numpy(self.rubbleMinerController.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            ore_action_mask = (
                th.from_numpy(self.oreMinerController.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )

            # SB3 doesn't support invalid action masking. So we do it ourselves here
            iceFeatures = self.iceMinerPolicy.policy.features_extractor(ice_obs.unsqueeze(0))
            rubbleFeatures = self.rubbleMinerPolicy.policy.features_extractor(rubble_obs.unsqueeze(0))
            oreFeatures = self.oreMinerPolicy.policy.features_extractor(ore_obs.unsqueeze(0))

            icex = self.iceMinerPolicy.policy.mlp_extractor.shared_net(iceFeatures)
            rubblex = self.rubbleMinerPolicy.policy.mlp_extractor.shared_net(rubbleFeatures)
            orex = self.oreMinerPolicy.policy.mlp_extractor.shared_net(oreFeatures)

            iceLogits = self.iceMinerPolicy.policy.action_net(icex) # shape (1, N) where N=12 for the default controller
            rubbleLogits = self.rubbleMinerPolicy.policy.action_net(rubblex)
            oreLogits = self.oreMinerPolicy.policy.action_net(orex)

            iceLogits[~ice_action_mask] = -1e8 # mask out invalid actions
            rubbleLogits[~rubble_action_mask] = -1e8
            oreLogits[~ore_action_mask] = -1e8

            iceDist = th.distributions.Categorical(logits=iceLogits)
            rubbleDist = th.distributions.Categorical(logits=rubbleLogits)
            oreDist = th.distributions.Categorical(logits=oreLogits)

            iceActions = iceDist.sample().cpu().numpy() # shape (1, 1)
            rubbleActions = rubbleDist.sample().cpu().numpy()
            oreActions = oreDist.sample().cpu().numpy()

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        ice_lux_action = self.iceMinerController.action_to_lux_action(
            self.player, raw_obs, iceActions[0])
        rubblelux_action = self.rubbleMinerController.action_to_lux_action(
            self.player, raw_obs, rubbleActions[0])
        ore_lux_action = self.oreMinerController.action_to_lux_action(
            self.player, raw_obs, oreActions[0])

        # join different robot actions
        lux_action = {**ice_lux_action, **rubblelux_action, **ore_lux_action}

        lux_action = collisionHandler(lux_action, self.player, raw_obs)

        # commented code below adds watering lichen which can easily improve your agent

        for unit_id in factories.keys():
            factory = factories[unit_id]
            if 1000 - step < 250 and factory["cargo"]["water"] > 100:
                lux_action[unit_id] = 2 # water and grow lichen at the very end of the game
            elif 1000 - step < 400 and factory["cargo"]["water"] > 1000:
                lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
