import argparse
# import datetime
# import os
import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium.envs.registration import register
from parking_env import ParkingEnv
from stable_baselines3 import TD3 #, DDPG, PPO, SAC
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.logger import configure
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.noise import NormalActionNoise #, OrnsteinUhlenbeckActionNoise

# import moviepy.editor as mpy

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='parking_env', help='nombre del entorno')
parser.add_argument('--render', type=bool, default=True, help='visualizar entorno')
parser.add_argument('--seed', type=int, default=0, help='semilla de los generadores aleatorios (default: 0)')
parser.add_argument('--total_timesteps', type=int, default=int(1000), help='pasos totales de entrenamiento')
parser.add_argument('--save_freq', type=int, default=int(100), help='frecuencia de guardado del modelo')
parser.add_argument('--log_path', type=str, default='./log', help='ruta de logging')
parser.add_argument('--ckpt_path', type=str, default='', help='ruta de guardado del modelo')
parser.add_argument('--mode', type=str, default='maniobra parcial', choices=['maniobra completa', 'maniobra parcial'], help='mode')

args = parser.parse_args()

register(
    id='parking_env',
    entry_point='parking_env:ParkingEnv'
)

env = gym.make(
    args.env,
    render=args.render,
    record_video=True,
    base_path='.',
    mode=args.mode,
    manual=False,
)

env.reset(options={'build_env':True})

# TD3
n_actions = env.action_space.shape[-1]

# Ruido de exploración (gaussiano)
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.05 * np.ones(n_actions)
)

# Ruido de exploración (Ornstein Uhlenbeck)
# action_noise = OrnsteinUhlenbeckActionNoise(
#     mean=np.zeros(n_actions),   # La media es cero (ruido alrededor de la acción dada).
#     sigma=np.array([0.1, 0.1]), # Desviación estándar del ruido. Ajusta la exploración.
#     theta=0.0015,               # Velocidad de reversión a la media. Valores más altos = ruido menos persistente.
#     dt=1.0                      # Paso de tiempo. Debe coincidir aproximadamente con el control_timestep.
# )

model = TD3(
    "MlpPolicy",
    env,
    learning_starts=1000,       # Pasos aleatorios antes de aprender
    action_noise=action_noise,  # Ruido de exploración externo
    verbose=1,
)

# Reducir los pesos de la última capa del actor
with torch.no_grad():
    # En TD3 de SB3, mu es un nn.Sequential. 
    # El orden interno suele ser: [nn.Linear, nn.ReLU, ..., nn.Linear, nn.Tanh]
    mu_sequence = model.policy.actor.mu
    
    # Extraemos la capa lineal final buscando de atrás hacia adelante en la secuencia
    last_layer = None
    for module in reversed(list(mu_sequence.children())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break

    # Aplicamos la inicialización de pesos
    if last_layer is not None:
        torch.nn.init.uniform_(last_layer.weight, -3e-3, 3e-3)
        torch.nn.init.zeros_(last_layer.bias)
        print(f"¡Capa lineal final del Actor inicializada! Estructura: {last_layer}")
    else:
        print("Error crítico: No se encontró ninguna capa nn.Linear dentro de model.policy.actor.mu")

# variables para logging
# time = datetime.datetime.strftime(datetime.datetime.now(), '%m%d_%H%M')
# log_path = os.path.join(args.log_path, f'TD3_{args.mode}_{time}')

# # if not args.ckpt_path:
# #     args.ckpt_path = os.path.join(args.log_path, f'td3_agent')
# ckpt_path = os.path.join(log_path, f'td3_agent')

# # logger = configure(args.log_path, ["stdout", "csv", "tensorboard"])
# logger = configure(log_path, ["stdout", "csv"])
# model.set_logger(logger)

# checkpoint_callback = CheckpointCallback(
#     save_freq=args.save_freq,
#     save_path=log_path,
#     name_prefix='td3_agent'
# )

# env.unwrapped.iniciar_grabacion_opencv(filename="estacionamiento paralelo - OpenCV.mp4", fps=30, frames_por_step=4)
env.unwrapped.iniciar_grabacion_moviepy(filename="estacionamiento paralelo - MoviePy.mp4", fps=30, frames_por_step=4)

args.mode = 'maniobra parcial'
args.total_timesteps = 1000

env.reset(options={"mode": args.mode})

model.learn(
    total_timesteps=args.total_timesteps,
    # callback=checkpoint_callback,
    log_interval=1,
    # reset_num_timesteps=False
)

args.mode = 'maniobra completa'
args.total_timesteps = 1000

env.reset(options={"mode": args.mode})

model.learn(
    total_timesteps=args.total_timesteps,
    # callback=checkpoint_callback,
    log_interval=1,
    # reset_num_timesteps=False
)

for _ in range(3):

    args.mode = 'maniobra parcial'
    args.total_timesteps = 500

    env.reset(options={"mode": args.mode})

    model.learn(
        total_timesteps=args.total_timesteps,
        # callback=checkpoint_callback,
        log_interval=1,
        reset_num_timesteps=False
    )

    args.mode = 'maniobra completa'
    args.total_timesteps = 5000

    env.reset(options={"mode": args.mode})

    model.learn(
        total_timesteps=args.total_timesteps,
        # callback=checkpoint_callback,
        log_interval=1,
        reset_num_timesteps=False
    )

# env.unwrapped.detener_grabacion_opencv()
env.unwrapped.detener_grabacion_moviepy()
env.close()

# '''video = mpy.VideoFileClip(filepath).subclip(0.02)
# video.write_videofile(filepath.replace('.mp4', '2.mp4'))'''
