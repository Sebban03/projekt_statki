import numpy as np


class RandomMaskedAgent:
    def act(self, env, agent_name):
        mask = env.action_mask(agent_name)
        legal = np.where(mask == 1)[0]
        return int(np.random.choice(legal))