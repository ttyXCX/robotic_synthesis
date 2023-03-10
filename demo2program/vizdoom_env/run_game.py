from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import sleep
import numpy as np

from cv2 import resize, INTER_AREA
from tqdm import tqdm

from vizdoom_env import Vizdoom_env
from dsl.dsl_parse import parse as vizdoom_parse
from dsl.random_code_generator import DoomProgramGenerator
from dsl.vocab import VizDoomDSLVocab
from util import log
import vizdoom as vzd

world = Vizdoom_env(config="vizdoom_env/asset/default.cfg",
                    perception_type='simple')
                    
world.game.set_mode(vzd.Mode.SPECTATOR)
world.init_game()

episodes = 10
class DoomStateGenerator(object):
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.x_max = 64
        self.x_min = -480
        self.y_max = 480
        self.y_min = 64

    def gen_rand_pos(self):
        return [self.rng.randint(self.x_min, self.x_max),
                self.rng.randint(self.y_min, self.y_max)]

    def get_pos_keys(self):
        return ['player_pos', 'demon_pos', 'hellknight_pos',
                'revenant_pos', 'ammo_pos']

    # generate an initial env
    def generate_initial_state(self, min_ammo=4, max_ammo=5,
                               min_monster=4, max_monster=5):
        """ h is y, w is x
            s = [{"player_pos": [x, y], "monster_pos": [[x1, y1], [x2, y2]]}]
        """
        s = {}
        locs = []
        s["player_pos"] = self.gen_rand_pos()
        s["demon_pos"] = []
        s["hellknight_pos"] = []
        s["revenant_pos"] = []
        s["ammo_pos"] = []
        locs.append(s["player_pos"])

        ammo_count = self.rng.randint(min_ammo, max_ammo + 1)
        demon_count = self.rng.randint(min_monster, max_monster + 1)
        hellknight_count = self.rng.randint(min_monster, max_monster + 1)
        revenant_count = self.rng.randint(min_monster, max_monster + 1)
        while(revenant_count > 0):
            new_pos = self.gen_rand_pos()
            if new_pos not in locs:
                s["revenant_pos"].append(new_pos)
                locs.append(new_pos)
                revenant_count -= 1

        while(hellknight_count > 0):
            new_pos = self.gen_rand_pos()
            if new_pos not in locs:
                s["hellknight_pos"].append(new_pos)
                locs.append(new_pos)
                hellknight_count -= 1

        while(demon_count > 0):
            new_pos = self.gen_rand_pos()
            if new_pos not in locs:
                s["demon_pos"].append(new_pos)
                locs.append(new_pos)
                demon_count -= 1

        while(ammo_count > 0):
            new_pos = self.gen_rand_pos()
            if new_pos not in locs:
                s["ammo_pos"].append(new_pos)
                locs.append(new_pos)
                ammo_count -= 1
        return s

gen = DoomStateGenerator(seed=123)

for i in range(episodes):
        print("Episode #" + str(i + 1))

        world.new_episode(gen.generate_initial_state())
        while not world.game.is_episode_finished():
            state = world.game.get_state()

            world.game.advance_action()
            last_action = world.game.get_last_action()
            reward = world.game.get_last_reward()

            print("State #" + str(state.number))
            print("Game variables: ", state.game_variables)
            print("Action:", last_action)
            print("Reward:", reward)
            print("=====================")

        print("Episode finished!")
        print("Total reward:", world.game.get_total_reward())
        print("************************")
        sleep(2.0)