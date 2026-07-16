import argparse
import gymnasium as gym
from gymnasium.envs.registration import register
from parking_env import ParkingEnv
import pybullet as p
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="parking_env", help='nombre del entorno')
parser.add_argument('--render', type=bool, default=True, help='visualizar entorno')
parser.add_argument('--mode', type=str, default='maniobra completa', choices=['maniobra completa', 'maniobra parcial'], help='mode')

args = parser.parse_args()

register(
    id='parking_env',
    entry_point='parking_env:ParkingEnv'
)

env = gym.make(args.env,
               render=args.render,
               base_path='.',
               mode=args.mode,
               manual=True)

env.reset(options={'build_env':True})

action=np.zeros(2)
action_pending = False
key_released = True
while True:
    keys = p.getKeyboardEvents()
    for k, v in keys.items():
        if k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN) and key_released:
            action[1] = max(action[1] - 0.1, -1)
            action_pending = True
            key_released = False
        elif k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN) and key_released:
            action[1] = min(action[1] + 0.1, 1)
            action_pending = True
            key_released = False
        elif k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN) and key_released:
            action[0] = min(action[0] + 0.1, 1)
            action_pending = True
            key_released = False
        elif k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN) and key_released:
            action[0] = max(action[0] - 0.1, -1)
            action_pending = True
            key_released = False
        elif k == p.B3G_RETURN and (v & p.KEY_IS_DOWN) and key_released:
            action_pending = True
            key_released = False
        elif (k == p.B3G_LEFT_ARROW or k == p.B3G_RIGHT_ARROW or k == p.B3G_UP_ARROW or k == p.B3G_DOWN_ARROW or k == p.B3G_RETURN) and (v & p.KEY_WAS_RELEASED):
            key_released = True

            if action_pending:
                next_state, reward, done, _, _ = env.step(action)
                action_pending = False
