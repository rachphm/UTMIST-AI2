# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class SubmittedAgent(Agent): #13
      def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0
        self.start_pos = None
        self.opp_start_pos = None
        self.target_pos = None
        self.prev_opp_pos = None

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        #More vars
        state = self.obs_helper.get_section(obs, 'player_state')
        jumps_left = self.obs_helper.get_section(obs, 'player_jumps_left')
        opp_state = self.obs_helper.get_section(obs, 'opponent_state')
        opp_vel = self.obs_helper.get_section(obs, 'opponent_vel')
        damage = self.obs_helper.get_section(obs, 'player_damage')
        opp_damage = self.obs_helper.get_section(obs, 'opponent_damage')
        dist_from_opp = abs((pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2)**0.5
        weapon = self.obs_helper.get_section(obs, 'player_weapon_type') #[0] no weapon, [1] spear, [2] hammer
        keep_distance = 1.7 if weapon[0] == 0 else 2
        safe_area = 2 if opp_pos[0] > 6.5 or opp_pos[0] < -6.5 else 1.5
        spawner1 = self.obs_helper.get_section(obs, 'player_spawner_1') #[x, y, z] x y is position
        spawner2 = self.obs_helper.get_section(obs, 'player_spawner_2') #[x, y, z] x y is position
        spawner3 = self.obs_helper.get_section(obs, 'player_spawner_3')
        spawner4 = self.obs_helper.get_section(obs, 'player_spawner_4')
        spawners = [spawner1, spawner2, spawner3, spawner4]

        valid_spawners = []
        for s in spawners:
            if s[0] != 0:
                dist = abs(s[0] - pos[0])
                valid_spawners.append((dist, s))

        if valid_spawners:
            # pick the closest one
            valid_spawners.sort(key=lambda x: x[0])
            spawner = valid_spawners[0][1]
        else:
            spawner = [20, 0, 0]

        approaching = False
        if self.prev_opp_pos is not None:
            prev_dist = abs(self.prev_opp_pos[0] - pos[0])
            curr_dist = abs(opp_pos[0] - pos[0])

            # approaching if getting closer
            approaching = curr_dist < prev_dist - 0.05
        self.prev_opp_pos = opp_pos.copy()

        #states
        at_g1 = ( -7 < pos[0] < -2 ) and (pos[1] > 2)
        at_g2 = ( 2 < pos[0] < 7 ) and (pos[1] > 0)
        opp_at_g1 = ( -7 < opp_pos[0] < -1 ) and (opp_pos[1] > 2)
        opp_at_g2 = ( 2 < opp_pos[0] < 7 ) and (opp_pos[1] > 0)
        on_ground = self.obs_helper.get_section(obs, 'player_grounded')[0] == 1
        opp_on_ground = self.obs_helper.get_section(obs, 'opponent_grounded')[0] == 1
        opp_overhead = pos[1] - opp_pos[1] > 2.5 and abs(pos[0] - opp_pos[0]) < 1.5
        is_dodging = self.obs_helper.get_section(obs, 'player_dodge_timer')[0] != 0
        not_facing_oppR = (pos[0] > opp_pos[0] and self.obs_helper.get_section(obs, 'player_facing')[0] == 1)
        not_facing_oppL = (pos[0] < opp_pos[0] and self.obs_helper.get_section(obs, 'player_facing')[0] == 0)
        opp_attack = self.obs_helper.get_section(obs, 'opponent_move_type')[0] != 0
        opp_heavy_attack = self.obs_helper.get_section(obs, 'opponent_move_type')[0] == 5
        middle_zone = pos[1] > 4 and 0 < pos[0] < 2 or pos[1] > 3 and -2 < pos[0] < 0

        #Store starting pos
        if self.start_pos is None:
            self.start_pos = pos.copy() 
            print(f"Starting position recorded: {self.start_pos}")
        if self.opp_start_pos is None:
            self.opp_start_pos = opp_pos.copy() 
            print(f"Opponent starting position recorded: {self.opp_start_pos}")
        if self.target_pos is None:
            self.target_pos = [4, 0.85]
        
        if damage >= opp_damage:
            if opp_at_g1 and not opp_KO and self.time % 201 == 0:
                self.target_pos = [-4, 2.85] 
            elif opp_at_g2 and not opp_KO and self.time % 201 == 0:
                self.target_pos = [4, 0.85]

        print(middle_zone)

        if middle_zone:
            action = self.act_helper.press_keys(['a'])
            # if pos[1] > 3.3 and self.time % 8 == 0:
            #     action = self.act_helper.press_keys(['space'], action)
            if pos[0] < 1 and self.time % 2 == 0:
                action = self.act_helper.press_keys(['space'], action)

        elif not opp_KO:
            #If off the edge then come back
            if pos[0] > self.target_pos[0] + safe_area: #If not at opponent's ground
                action = self.act_helper.press_keys(['a'])
            elif pos[0] < self.target_pos[0] - safe_area: #If not at opponent's ground
                action = self.act_helper.press_keys(['d'])

            
            elif dist_from_opp < 5 and pos[1] + 2 > opp_pos[1] :
                # Head toward opponent
                if opp_pos[0] - pos[0] > keep_distance or not_facing_oppL:
                    action = self.act_helper.press_keys(['d'])
                elif pos[0] - opp_pos[0] > keep_distance or not_facing_oppR:
                    action = self.act_helper.press_keys(['a'])
                if pos[1] > opp_pos[1] and pos[1] > -4 and self.time%2 == 0:
                    action = self.act_helper.press_keys(['k'], action)
                    action = self.act_helper.press_keys(['space'], action)

                #Attack
                # self.target_pos[0] - 1.5 < pos[0] < self.target_pos[0] + 1.5 and
                if (dist_from_opp > 3 and approaching) and self.time%2 == 0:                    
                    action = self.act_helper.press_keys(['k'], action)
                else:
                    action = self.act_helper.press_keys(['j'], action)

                if opp_attack and not is_dodging: #TIS IS WORKING DO NOT TOUCH
                    action = self.act_helper.press_keys(['l'], action)

            #Jump
            if (pos[1] > self.target_pos[1]+0.7 or \
                pos[0] < self.target_pos[0] - 2.6 or \
                pos[0] > self.target_pos[0] + 2.6 #2.5 and 3 is working
                ) and self.time %10 == 0:
                action = self.act_helper.press_keys(['space'], action)
            if opp_heavy_attack:
                action = self.act_helper.press_keys(['space'], action)

            if pos[1] > self.target_pos[1] + 1 and self.time %4 == 0:
                action = self.act_helper.press_keys(['space'], action)

        elif opp_KO:
            if weapon[0] != 2 and spawner[0] != 20:
                if (spawner[0] > pos[0]):
                    action = self.act_helper.press_keys(['d'])
                elif (spawner[0] < pos[0]):
                    action = self.act_helper.press_keys(['a'])
            else:
                if pos[0] > self.target_pos[0] + safe_area: #If not at opponent's ground
                    action = self.act_helper.press_keys(['a'])
                elif pos[0] < self.target_pos[0] - safe_area: #If not at opponent's ground
                    action = self.act_helper.press_keys(['d'])

            if -3 < pos[0] < -1 and self.time % 2 == 0:
                action = self.act_helper.press_keys(['space'], action)

        # Pick up weapon if near
        if weapon[0] != 2 and abs(spawner[0] - pos[0]) < 1:
            action = self.act_helper.press_keys(['h'], action)

   
        #Speed fall if safe
        if (3 < pos[0] < 5 or -5 < pos[0] < -3) and not on_ground:
            action = self.act_helper.press_keys(['s'], action)

        return action
