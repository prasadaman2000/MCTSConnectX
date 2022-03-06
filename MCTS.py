# TODO create way to load a checkpoint file
# TODO figure out where to stick in a DNN

import numpy as np
import random
import json

# hacky solution for checkpointing
class MyEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__    


# class to store arbitrary action sets, such as possible moves in ConnectX
class ActionSet():

    # initializes an actionset by just taking in a list of possible actions
    def __init__(self, actions: list):
        self.actions = actions
        
        # initial number of samples in the action set needs to be non-zero.
        # initialize total_samples to len(actions) in order to create a uniform distribution over all actions
        self.total_samples = len(actions)

        # same reason as above, initialize the number of wins per action to 1
        self.num_wins = [1] * len(actions)

    # chooses an action based on the probability distribution
    def action_sample(self):
        # generate the probability distribution. total != 1 since there is a probability of losing
        prob_dist = np.array(self.num_wins) / self.total_samples

        # need outer random.choice because random.choices returns some sort of list TODO figure out why
        return random.choice(random.choices(self.actions, prob_dist))

    # updates the probability distibution
    # win is either 1 or 0, 1 for win, 0 for loss
    def update(self, action, win):
        self.total_samples += 1
        action_idx = self.actions.index(action)
        self.num_wins[action_idx] += win


# the thing that will train the MCTS agent
class MCTS_Trainer:
    def __init__(self):
        self.state_action_map = {}
    
    def train(self, env, trainer, num_eps=10):
        wins = 0
        total = 0

        # the number of columns in the environment (default is 7 in connect4)
        columns = env.configuration.columns

        # loop for each game that needs to be played during training
        for ep in range(1, num_eps + 1):

            #game reset
            observation = trainer.reset()
            reward = 0
            observations = []

            # game loop
            while not env.done:
                # get current game state and store as hashable data
                board = tuple(observation.board)
                board_key = str(board)

                # get the action set for this particular state
                actionset = None
                if board_key in self.state_action_map:
                    actionset = self.state_action_map[board_key]
                else:
                    # if the state is new, add it to the dict of visited states
                    possible_actions = [c for c in range(columns) if board[c] == 0]
                    actionset = ActionSet(possible_actions)
                    self.state_action_map[board_key] = actionset

                # choose an action
                action = actionset.action_sample()
                observations.append((board_key, action))

                # do the action and get the next state
                observation, reward, _, _ = trainer.step(action)

            # loss reward is -1 but needs to be 0 to be compatible with ActionSet
            # win reward is 1
            if reward == -1:
                reward = 0
            
            # back propogation
            for board_key, action in observations:
                self.state_action_map[board_key].update(action, reward)
            
            wins += reward
            total += 1

            # some logging
            if ep % 100 == 0:
                print(f"End of ep {ep}: win percentage: {100 * wins / total}%")
        

        # saves self.state_action_map in a dict since it's the only thing necessary
        #      for a checkpoint. Uses MyEncoder class to successfully serialize ActionSet
        fname = f"table_win_{wins}_in_{total}.json"
        with open(fname, "w") as f:
            json.dump(self.state_action_map, f, cls=MyEncoder, indent=4)
