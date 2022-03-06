from kaggle_environments import make

# make the environment
env = make("connectx", debug=True)

# train against built in negamax agent, with training agent as first player
trainer = env.train([None, 'negamax'])

# use class from MCTS.py
from MCTS import MCTS_Trainer
t = MCTS_Trainer()
t.train(env, trainer, num_eps=100000)
