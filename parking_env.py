import matplotlib.pyplot as plt

import os
import random
import time

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from collections import defaultdict

import cv2
import moviepy

from typing import List #, Optional


class ParkingEnv(gym.Env):
    metadata = {'render_modes': ['human','rgb_array']}

    def __init__(
            self,
            render=False,
            base_path=os.getcwd(),
            mode='maniobra completa',
            manual=False,
            record_video=False,
        ):
        """
        :param render: 
        :param base_path: 
        :param mode: 
        :param manual: 
        :param render_video: 
        """

        self.base_path = base_path
        self.manual = manual
        self.mode = mode
        assert self.mode in ['maniobra completa', 'maniobra parcial']

        self.car = None
        self.done = False
        self.success = False
        self.goal = None

        self.ground = None
        self.wall_parking_rear = None
        self.wall_parking_front = None
        self.wall_parking_left = None
        self.wall_lane_front_left = None
        self.wall_lane_rear_left = None
        self.wall_lane_rear_right = None
        self.wall_lane_center_right = None
        self.wall_lane_front_right = None

        self.max_speed = 0.1
        self.max_steering_angle = np.pi/5
        self.max_pos_error_x = np.abs(0.31 - 0.00) + 0.15
        self.max_pos_error_y = np.abs(0.36 - 0.57) + 0.15

        self.scales = np.array([
            self.max_pos_error_x,
            self.max_pos_error_y,
            1.0,
            1.0,
            self.max_speed,
            self.max_speed
        ], dtype=np.float32)

        obs_low = -np.ones(len(self.scales), dtype=np.float32)
        obs_high = np.ones(len(self.scales), dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = spaces.Box(np.float32(-1), np.float32(1), (2,))
        
        self.target_orientation = None  # yaw
        self.start_orientation = None   # [roll, pitch, yaw]

        self.action_steps = 240

        self.step_threshold = 1000 // (self.max_speed * self.action_steps)

        self.step_cnt = 0
        self.train_step_cnt = 0

        # Almacenamiento de componentes de recompensa
        self.historial_componentes = defaultdict(list)
        
        if self.manual:
            # Configuración de la figura para graficar en tiempo real
            self.fig = None
            self.graficar_en_tiempo_real()

        if render:
            self.client = p.connect(p.GUI, options="--width=640 --height=480")
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.rectangulo_ids = []  # Para guardar los objetos del rectángulo

        # --- Configuración de renderizado y grabación ---
        self.grabando = False
        if record_video:
            self.render_mode = "rgb_array"

            self.carpeta = "videos/"
            self.video_filename = "estacionamiento paralelo.mp4"
            self.fps = 30
            self.frames_por_step = 4
            # self.step_counter = 0
            self.frame_counter = 0

            os.makedirs(self.carpeta, exist_ok=True)

            # OpenCV
            self.video_writer = None

            # MoviePy
            self.frames: List[np.ndarray] = []
            
            self.width = 640
            self.height = 480

            self.camera_view_matrix = None
            self.camera_proj_matrix = None

            # Configurar cámara sólo si render_mode es rgb_array
            self.setup_camera()
        elif render: # sólo GUI
            self.render_mode = "human"
        else:
            self.render_mode = None
  
    def crear_rectangulo_suelo(self):
        """Crea un rectángulo estático en el suelo usando objetos visuales"""
        
        ancho = 0.206
        largo = 0.15
        color = [1, 1, 1, 1]
        
        # Rectángulo sólido
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[ancho/2, largo/2, 0.000]
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[ancho/2, largo/2, 0.010],
            rgbaColor=color
        )
        
        rect_id = p.createMultiBody(
            baseMass=0,  # Estático
            # baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.006, 0.57, 0.000],
            baseOrientation=[0, 0, 0, 1] # cuaternión canónico
        )
        
        self.rectangulo_ids.append(rect_id)
        
    def setup_camera(self):
        """Configura la matriz de proyección para renderizado off-screen"""
    
        # Matrices fijas de cámara
        self.camera_proj_matrix = p.computeProjectionMatrixFOV(
            fov=20.0, #60,
            aspect= self.width / self.height,
            nearVal=0.1,
            farVal=100.0
        )
        self.camera_view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0.49, 0],
            distance=2, #0.75,
            yaw=0,
            pitch=-60,
            roll=0,
            upAxisIndex=2
        )

    def render(self):
        """
        :param mode: 
        """
        # Si estamos en modo human o no especificado, devolver None
        if self.render_mode != "rgb_array":
            return None
        
        # Si estamos en modo rgb_array, capturar imagen
        img = p.getCameraImage(
            self.width, 
            self.height, 
            self.camera_view_matrix,
            self.camera_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
            # renderer=p.ER_TINY_RENDERER
        )
        
        # Extraer RGB (img[2] contiene los datos de píxeles)
        rgb = np.reshape(img[2], (self.height, self.width, 4))[:, :, :3]
        rgb = np.uint8(rgb)  # Asegurar tipo de dato correcto
        
        return rgb

# Funciones de grabación de video con OpenCV ###################################
    def iniciar_grabacion_opencv(self, filename="estacionamiento paralelo.mp4", fps=30, frames_por_step=4):
        """Inicia la grabación de video"""
        
        # Si estamos en modo human o no especificado, salir
        if self.render_mode != "rgb_array":
            return

        self.video_filename = filename
        video_path = os.path.join(self.carpeta, self.video_filename)
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (self.width, self.height)
        )
        self.frames_por_step = frames_por_step
        # self.step_counter = 0
        self.frame_counter = 0
        self.grabando = True
        print(f"✅ Grabación iniciada: {filename}")
        
    def capturar_frame_opencv(self):
        """Captura un frame usando render() y lo guarda en el video"""
        
        if not self.grabando or self.video_writer is None:
            return
            
        rgb = self.render()
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Agregar información al frame
        # info_text = f"Step: {self.step_counter} | Frame: {self.frame_counter}"
        # cv2.putText(rgb_bgr, info_text, (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self.video_writer.write(np.uint8(rgb_bgr))
        self.frame_counter += 1

    def detener_grabacion_opencv(self):
        """Detiene y guarda el video"""
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.grabando = False
            print(f"✅ Video guardado: {self.video_filename} ({self.frame_counter} frames)")
            self.video_writer = None
################################################################################

# Funciones de grabación de video con MoviePy ##################################
    def iniciar_grabacion_moviepy(self, filename="estacionamiento paralelo.mp4", fps=30, frames_por_step=4):
        """Inicia la grabación (limpia frames anteriores)"""

        # Si estamos en modo human o no especificado, salir
        if self.render_mode != "rgb_array":
            return

        self.video_filename = filename
        self.fps = fps
        self.frames = []
        self.frames_por_step = frames_por_step
        self.grabando = True
        print(f"🎬 Grabación iniciada: {filename}")
    
    def capturar_frame_moviepy(self):
        """Captura un frame usando render() y lo almacena en memoria"""

        if not self.grabando:
            return
        
        rgb = self.render()
        self.frames.append(rgb)
    
    def detener_grabacion_moviepy(self):
        """Detiene la grabación y genera el video con MoviePy"""
        
        if not self.grabando:
            return
        
        self.grabando = False
        
        if len(self.frames) == 0:
            print("⚠️ No se capturaron frames, video no generado.")
            return
        
        print(f"📹 Generando video con {len(self.frames)} frames...")
        
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        
        video_path = os.path.join(self.carpeta, self.video_filename)
        clip = ImageSequenceClip(self.frames, fps=self.fps)
        clip.write_videofile(video_path, verbose=False, logger=None)
        
        print(f"✅ Video guardado: {video_path}")
        self.frames = []
################################################################################

    def reset(self, seed=None, options=None):
        """
        """
        if isinstance(options, dict) and 'mode' in options:
            self.mode = options['mode']
            assert self.mode in ['maniobra completa', 'maniobra parcial']

        if self.mode == 'maniobra completa':
            basePosition = [0.31, 0.36, 0.05]
            self.goal = np.array([0, 0.57])
            self.start_orientation = [0, 0, 0]
            self.target_orientation = 0
        elif self.mode == 'maniobra parcial':
            dx = (2*np.random.rand()-1) * 0.02
            dy = (2*np.random.rand()-1) * 0.01
            dtheta = (2*np.random.rand()-1) * np.pi * 5/180
            basePosition = [-0.07 + dx, 0.57 + dy, 0.05]
            self.goal = np.array([0, 0.57])
            self.start_orientation = [0, 0, 0 + dtheta]
            self.target_orientation = 0

        if isinstance(options, dict) and 'build_env' in options and options['build_env']:

            p.resetSimulation(self.client)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=0,
                cameraPitch=-89,
                cameraTargetPosition=[0, 0.49, 0]
            )
            p.setGravity(0, 0, -10)

            # Configurar parámetros de física general
            p.setPhysicsEngineParameter(
                numSolverIterations=100,   # Más iteraciones = mejor convergencia
                # numSubSteps=4,             # Más sub-pasos
                # contactBreakingThreshold=0.0001,
                # deterministicOverlapPairs=1
            )

            self.ground = p.loadURDF(
                os.path.join(
                    self.base_path,
                    "3Dmodels/ground.SLDPRT/urdf/ground.SLDPRT.urdf"),
                basePosition=[0, 0, 0.005],
                useFixedBase=True
            )

            # Configurar parámetros del suelo
            p.changeDynamics(
                self.ground, 
                -1,                     # -1 para el link base
                # mass=0,                 # lo hace estático
                lateralFriction=1.0,    # Fricción lateral
                spinningFriction=0.0,   # Fricción contra rotación
                rollingFriction=0.0,    # Resistencia a rodar
                # restitution=0.1,        # Elasticidad (bajo para suelo)
                # contactStiffness=1e7,   # Rigidez del contacto
                # contactDamping=1e3      # Amortiguación del contacto
            )

            self.wall_parking_rear = p.loadURDF(
                os.path.join(
                    self.base_path,
                    "3Dmodels/side_boundary.SLDPRT/urdf/side_boundary.SLDPRT.urdf"),
                basePosition=[-0.23, 0.73, 0.01],
                useFixedBase=True
            )
            self.wall_parking_front = p.loadURDF(
                os.path.join(
                    self.base_path,
                    "3Dmodels/side_boundary.SLDPRT/urdf/side_boundary.SLDPRT.urdf"),
                basePosition=[0.21, 0.73, 0.01],
                useFixedBase=True
            )
            self.wall_parking_left = p.loadURDF(
                os.path.join(
                    self.base_path,
                    "3Dmodels/front_boundary.SLDPRT/urdf/front_boundary.SLDPRT.urdf"),
                basePosition=[-0.21, 0.73, 0.01],
                useFixedBase=True
            )
            self.wall_lane_front_left = p.loadURDF(
                os.path.join(
                    self.base_path,
                    "3Dmodels/front_boundary.SLDPRT/urdf/front_boundary.SLDPRT.urdf"),
                basePosition=[0.21, 0.49, 0.01],
                useFixedBase=True
            )
            self.wall_lane_rear_left = p.loadURDF(
                os.path.join(
                    self.base_path,
                    "3Dmodels/front_boundary.SLDPRT/urdf/front_boundary.SLDPRT.urdf"),
                basePosition=[-0.63, 0.49, 0.01],
                useFixedBase=True
            )
            self.wall_lane_rear_right = p.loadURDF(
                os.path.join(
                    self.base_path,
                    "3Dmodels/front_boundary.SLDPRT/urdf/front_boundary.SLDPRT.urdf"),
                basePosition=[-0.63, 0.135, 0.01],
                useFixedBase=True
            )
            self.wall_lane_center_right = p.loadURDF(
                os.path.join(
                    self.base_path,
                    "3Dmodels/front_boundary.SLDPRT/urdf/front_boundary.SLDPRT.urdf"),
                basePosition=[-0.21, 0.135, 0.01],
                useFixedBase=True
            )
            self.wall_lane_front_right = p.loadURDF(
                os.path.join(
                    self.base_path,
                    "3Dmodels/front_boundary.SLDPRT/urdf/front_boundary.SLDPRT.urdf"),
                basePosition=[0.21, 0.135, 0.01],
                useFixedBase=True
            )
            
            # Perímetro rectangular objetivo
            # p.addUserDebugLine([-0.097, 0.57-0.075, 0.01], [-0.097, 0.57+0.075, 0.01], [0.75, 0.75, 0.75], 1)
            # p.addUserDebugLine([0.109, 0.57-0.075, 0.01], [0.109, 0.57+0.075, 0.01], [0.75, 0.75, 0.75], 1)
            # p.addUserDebugLine([-0.097, 0.57-0.075, 0.01], [0.109, 0.57-0.075, 0.01], [0.75, 0.75, 0.75], 1)
            # p.addUserDebugLine([-0.097, 0.57+0.075, 0.01], [0.109, 0.57+0.075, 0.01], [0.75, 0.75, 0.75], 1)

            # Objeto rectangular para marcar el objetivo
            self.crear_rectangulo_suelo()

            self.t = Car(
                self.client,
                basePosition=basePosition,
                baseOrientationEuler=self.start_orientation,
                max_speed=self.max_speed,
                max_steering_angle=self.max_steering_angle,
                action_steps=self.action_steps,
                base_path=self.base_path)
            self.car = self.t.car
        
        self.t.reiniciar_estado_vehiculo(basePosition, self.start_orientation)

        observation = self.get_obs()

        self.en_zona1 = False
        self.en_zona2 = False

        self.reward_xya_prev = np.zeros(3) #este valor no importa, se define para poder llamar a compute_reward y actualizarlo con el valor correcto
        self.compute_reward(observation, np.zeros(2))

        self.step_cnt = 0

        observation /= self.scales

        return observation, {}

    def apply_action(self, action):
        """
        """

        for sim_step in range(self.action_steps):

            self.t.car_sim_step(action, sim_step)
            
            # Si estamos grabando capturar múltiples frames durante el step (simulación intermedia)
            if self.grabando and (sim_step + 1) % (self.action_steps // self.frames_por_step) == 0:
                # self.capturar_frame_opencv()
                self.capturar_frame_moviepy()

    def compute_reward(self, observation, action=None):

        # PENALIZACIÓN POR COLISIÓN
        if self.judge_collision() == 1:
            reward = -10.0
            self.done = True
            return reward
        if self.judge_collision() == 2:
            reward = -20.0
            self.done = True
            return reward
        
        pos_error_xy = observation[:2]
        orientation_rel_xy = observation[2:4]

        # PENALIZACIÓN POR LÍMITE "X" EXCEDIDO
        if np.abs(pos_error_xy[0]) > self.max_pos_error_x:
            reward = -20.0
            self.done = True
            return reward

        # RECOMPENSA DOMINANTE POR ÉXITO
        if (np.abs(pos_error_xy[0]) < 0.03 and
            np.abs(pos_error_xy[1]) < 0.02 and
            orientation_rel_xy[0] > np.cos(np.pi * 15 / 180)):
            reward = 20.0 #10.0
            self.done = True
            self.success = True
            return reward

        # CAMBIO DE REFERENCIA DE POSICIÓN
        c = orientation_rel_xy[0]
        s = orientation_rel_xy[1]
        rot_2d = np.array([[c, s], [-s, c]])
        # r = np.array([-0.097, 0.075])     # esquina trasera izquierda
        # r = np.array([-0.097, 0.0])       # centro trasero
        r = np.array([-0.097, -0.075])      # esquina trasera derecha
        # r = np.array([-0.064, 0.075])     # extremo izquierdo del eje trasero
        # r = np.array([-0.064, 0.0])       # centro del eje trasero
        # r = np.array([-0.064, -0.075])    # extremo derecho del eje trasero
        pos_error_xy = pos_error_xy + r @ rot_2d - r
        # p.addUserDebugLine([0.00+r[0], 0.57+r[1], 0.01], [0.00+r[0]+pos_error_xy[0], 0.57+r[1]+pos_error_xy[1], 0.01], [0.75, 0.75, 0.75], 1)#, 0.5)

        # PENALIZACIÓN POR LÍMITE "Y" EXCEDIDO
        if np.abs(pos_error_xy[1]) > self.max_pos_error_y:
            reward = -20.0
            self.done = True
            return reward

        # RECOMPENSA COMBINADA

        distance_reward_x = np.clip((self.max_pos_error_x - np.abs(pos_error_xy[0])) / (self.max_pos_error_x), 0.0, 1.0)
        distance_reward_y = np.clip((self.max_pos_error_y - np.abs(pos_error_xy[1])) / (self.max_pos_error_y), 0.0, 1.0)

        theta = np.arctan2(orientation_rel_xy[1], orientation_rel_xy[0])
        angle_reward = np.clip((np.pi - np.abs(theta)) / np.pi, 0.0, 1.0)

        # Función de recompensa densa en función de la distancia y la orientación relativa al objetivo
        # reward = -np.power(
        #     np.dot(
        #         np.array([1-distance_reward_x, 1-distance_reward_y, 1-angle_reward]),
        #         np.array([0.10, 0.50, 0.40]),
        #     ),
        #     0.5,
        # )
        distance_reward_x *= 0.10
        distance_reward_y *= 0.50
        angle_reward *= 0.40
        reward = -np.power(
            1.0 - (
                + distance_reward_x
                + distance_reward_y
                + angle_reward
            ),
            0.5
        )
        
        # Función de recompensa densa en función del delta de la distancia y de la orientación relativa al objetivo
        # reward_xya = np.array([distance_reward_x, distance_reward_y, angle_reward])
        # delta_reward_xya = reward_xya - self.reward_xya_prev
        # reward += 10.0 * np.dot(
        #     delta_reward_xya,
        #     np.array([0.10, 0.50, 0.40])
        # )
        # self.reward_xya_prev = reward_xya

        # Reward shaping con penalizaciones y bonificaciones
        # en_zona1 = np.abs(pos_error_xy[1]) < 0.025 and orientation_rel_xy[0] > np.cos(np.pi * 30 / 180) # +/-20°
        # en_zona2 = np.abs(pos_error_xy[1]) < 0.020 and orientation_rel_xy[0] > np.cos(np.pi * 15 / 180) # +/-10°

        # bonus = 0
        # penalty = 0
        # if not en_zona1:
        #     if self.en_zona1:
        #         print('fuera de zona 1')
        #         self.en_zona1 = False
        #         penalty += 10.0
        #     # if action == 2:
        #     if action[0] > 0:
        #         penalty += 0.1 #* action[0]
        #     # elif action[0] < 0:
        #     #     bonus += 0.1 #* -action[0]
        # else: # en zona 1
        #     if not self.en_zona1:
        #         print('en zona 1')
        #         self.en_zona1 = True
        #         bonus += 10.0
        #     # bonus += 10.0
        #     if action[0] > 0:
        #         bonus += 0.1 #* action[0]
        #     if not en_zona2:
        #         if self.en_zona2:
        #             print('fuera de zona 2')
        #             self.en_zona2 = False
        #             penalty += 10.0
        #     else: # en zona 2
        #         if not self.en_zona2:
        #             print('en zona 2')
        #             self.en_zona2 = True
        #             bonus += 10.0
        #         # bonus += 10.0
        #         # if action != 2:
        #         if action[0] < 0:
        #             penalty += 0.1 #* -action[0]
        #         # elif action[0] > 0:
        #         #     bonus += 0.1 #* action[0]

        # reward += (bonus - penalty)

        # Almacena las componentes
        self.historial_componentes['distance_reward_x'].append(distance_reward_x)
        self.historial_componentes['distance_reward_y'].append(distance_reward_y)
        self.historial_componentes['angle_reward'].append(angle_reward)
        self.historial_componentes['reward'].append(reward)

        return reward

    def judge_collision(self):
        """
        """

        if p.getContactPoints(self.car, self.wall_parking_rear): return 1
        if p.getContactPoints(self.car, self.wall_parking_front): return 1
        if p.getContactPoints(self.car, self.wall_parking_left): return 1
        if p.getContactPoints(self.car, self.wall_lane_front_left): return 2
        if p.getContactPoints(self.car, self.wall_lane_rear_left): return 1
        if p.getContactPoints(self.car, self.wall_lane_rear_right): return 2
        if p.getContactPoints(self.car, self.wall_lane_center_right): return 2
        if p.getContactPoints(self.car, self.wall_lane_front_right): return 2

        return 0

    def step(self, action):
        """
        :param action:
        :return: observation, reward, terminated, truncated, info
        """
        self.apply_action(action)

        observation = self.get_obs()

        self.done = False
        self.success = False

        reward = self.compute_reward(observation, action)

        if self.manual:
            distance = np.linalg.norm(observation[:2])
            orientation_error = np.linalg.norm(observation[2:4] - np.array([1, 0]))

            print(f's:{self.train_step_cnt}|a:[{action[0]:.1f} {action[1]:.1f}]|d:{distance:.3f}|oe:{orientation_error:.3f}|r:{reward:.2f}')

            # Graficar después de cada paso
            self.graficar_en_tiempo_real()

        self.step_cnt += 1
        if self.step_cnt > self.step_threshold:
            self.done = True

        if self.done:
            self.train_step_cnt += self.step_cnt
            self.step_cnt = 0

        info = {'is_success': self.success}

        observation /= self.scales
        # print(f'observation= {observation}')

        return observation, reward, self.done, False, info

    def graficar_en_tiempo_real(self):
        """Grafica en tiempo real actualizando la misma figura"""

        # Inicializar figura si no existe (SOLO UNA VEZ)
        if self.fig is None:
            print("Creando figura...")
            plt.ion()
            
            # Crear figura
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6))
            
            # Configurar posición
            try:
                self.fig.canvas.manager.window.wm_geometry('+100+100')
            except:
                pass
            
            # Inicializar líneas
            self.lines = {}
            colores = ['blue', 'orange', 'green', 'red']
            nombres = ['distance_reward_x', 'distance_reward_y', 'angle_reward', 'reward']
            
            for idx, nombre in enumerate(nombres):
                line, = self.ax1.plot(
                    [], [], 
                    label=nombre, 
                    color=colores[idx],
                    linewidth=2 if nombre == 'reward' else 1,
                    alpha=0.7 if nombre != 'reward' else 1.0
                )
                self.lines[nombre] = line
            
            self.ax1.set_xlabel('Paso')
            self.ax1.set_ylabel('Valor de recompensa')
            self.ax1.set_title('Componentes de recompensa en tiempo real')
            self.ax1.legend(loc='upper right')
            self.ax1.grid(True, alpha=0.3)
            
            # Línea para la recompensa acumulada
            self.line_acumulada, = self.ax2.plot([], [], 'r-', linewidth=2, label='Total acumulada')
            self.ax2.set_xlabel('Paso')
            self.ax2.set_ylabel('Recompensa acumulada')
            self.ax2.set_title('Recompensa total acumulada')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend(loc='upper left')
            
            # MOSTRAR LA FIGURA (SOLO UNA VEZ)
            self.fig.show()
            self.fig.canvas.draw()
            
            # SOLO UNA VEZ para renderizar
            plt.pause(0.001)
            
            print("Figura creada exitosamente")
            return  # Salir para no actualizar con datos vacíos
        
        # Verificar que hay datos
        if len(self.historial_componentes['reward']) == 0:
            return
        
        # Actualizar datos
        pasos = list(range(len(self.historial_componentes['reward'])))
        
        for nombre, line in self.lines.items():
            if nombre in self.historial_componentes:
                line.set_data(pasos, self.historial_componentes[nombre])
        
        # Actualizar la recompensa acumulada
        acumulada = np.cumsum(self.historial_componentes['reward'])
        self.line_acumulada.set_data(pasos, acumulada)
        
        # Ajustar límites
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()
        
        # Redibujar
        self.fig.canvas.draw_idle()  # draw_idle() no roba foco
        self.fig.canvas.flush_events()  # flush_events() no roba foco

    def seed(self, seed=None):
        """
        :param seed:
        :return: [seed]
        """

        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        """
        """
        if self.client is not None:
            p.disconnect(self.client)

    def get_obs(self):
        """
        """
        pos, orn = p.getBasePositionAndOrientation(self.car)
        yaw = p.getEulerFromQuaternion(orn)[2]
        vel, ang_vel = p.getBaseVelocity(self.car)

        dx = pos[0] - self.goal[0]
        dy = pos[1] - self.goal[1]
        dtheta = yaw - self.target_orientation
        
        # Estado relativo al objetivo
        car_obs = np.array([
            dx,             # ΔX
            dy,             # ΔY
            np.cos(dtheta), # cos(Δθ)
            np.sin(dtheta), # sin(Δθ)
            vel[0],         # Vx
            vel[1],         # Vy
            # ang_vel[2]    # w
        ], dtype=np.float32)

        return car_obs


class Car:
    def __init__(
            self,
            client,
            basePosition=[0, 0, 0.05],
            baseOrientationEuler=[0, 0, np.pi / 2],
            max_speed=0.1,
            max_steering_angle=np.pi/4,
            max_force=100,
            action_steps=None,
            base_path=os.getcwd()
        ):
        """
        :param client: pybullet client
        :param basePosition: 
        :param baseOrientationEuler: 
        :param max_velocity: 
        :param max_force: 
        :param action_steps: 

        Car URDF original:
        Joint 0: lb_wheel
        Joint 1: rb_wheel
        Joint 2: servo
        Joint 3: turn_l
        Joint 4: lf_wheel
        Joint 5: link_l
        Joint 6: link_ll
        Joint 7: turn_r
        Joint 8: rf_wheel
        Joint 9: link_rr
        Joint 10: camera

        Car URDF modificado:
        Joint 0: lb_wheel
        Joint 1: rb_wheel
        Joint 2: servo
        Joint 3: turn_l
        Joint 4: lf_wheel
        Joint 5: turn_r
        Joint 6: rf_wheel
        Joint 7: camera
        """

        self.client = client
        self.base_path = base_path
        self.car = p.loadURDF(
            os.path.join(
                self.base_path,
                "3Dmodels/car.SLDASM/urdf/car.SLDASM.urdf"
            ),
            basePosition=basePosition,
            baseOrientation=p.getQuaternionFromEuler(baseOrientationEuler)
        )

        # Masa chasis
        p.changeDynamics(
            self.car,
            -1,
            mass=0.5    # Chasis
        )

        self.drive_joints = [0, 1, 4, 6]
        self.steering_joints = [3, 5]

        # Configurar parámetros de las ruedas
        for wheel_idx in self.drive_joints:
            p.changeDynamics(
                self.car,
                wheel_idx,
                mass=0.010,             # Escalar masa (1% al 2.5% del peso total del vehículo)
                lateralFriction=2.0,    # Buen agarre pero no excesivo para evitar volcar
                rollingFriction=0.0001, # Casi nula para que no actúe como freno de mano
                spinningFriction=0.005, # Valor micro
                frictionAnchor=1,       # Crucial para mantener estabilidad en pesos bajos
                # contactStiffness=1e7,
                # contactDamping=1e3,
                # restitution=0.1
            )

        self.max_force = max_force
        self.action_steps = action_steps
        self.dt = 1/240                 # Frecuencia de PyBullet

        self.speed_max = max_speed
        self.speed_max_pos = self.speed_max / 2
        self.speed_max_neg = self.speed_max
        self.speed_ini = 0

        self.steering_angle_max = max_steering_angle
        self.steering_angle_ini = 0

        # Desactivar motores por defecto en todas las articulaciones
        for joint in self.drive_joints + self.steering_joints:
            p.setJointMotorControl2(
                self.car,
                joint,
                p.VELOCITY_CONTROL,
                # targetVelocity=0,
                force=0             # asegurar force=0
            )

    def car_sim_step(self, action, sim_step):
        """
        2 acciones continuas (DDPG, TD3, SAC)
        """
        if sim_step == 0:
        
            if action[0] >= 0:
                self.speed_fin = action[0] * self.speed_max_pos
            else:
                self.speed_fin = action[0] * self.speed_max_neg

            self.steering_angle_fin = action[1] * self.steering_angle_max

        t = (sim_step + 1) / self.action_steps

        speed = (1 - t) * self.speed_ini + t * self.speed_fin

        ang_vel = speed / 0.0325
        
        p.setJointMotorControl2(
            bodyUniqueId=self.car,
            jointIndex=self.drive_joints[0],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=ang_vel,
            force=0.03                      # Torque balanceado para 0.5kg a baja velocidad (0.02 a 0.04 Nm)
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.car,
            jointIndex=self.drive_joints[1],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-ang_vel,
            force=0.03                      # Torque balanceado para 0.5kg a baja velocidad (0.02 a 0.04 Nm)
        )

        steering_angle = (1 - t) * self.steering_angle_ini + t * self.steering_angle_fin

        p.setJointMotorControl2(
            bodyUniqueId=self.car,
            jointIndex=self.steering_joints[0],
            controlMode=p.POSITION_CONTROL,
            targetPosition=steering_angle,
            force=0.4,                          # Suficiente para levantar la esquina trasera para 0.5kg (0.3 a 0.5 Nm)
            positionGain=0.5, #0.03,
            velocityGain=1.0
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.car,
            jointIndex=self.steering_joints[1],
            controlMode=p.POSITION_CONTROL,
            targetPosition=-steering_angle,
            force=0.4,                          # Suficiente para levantar la esquina trasera para 0.5kg (0.3 a 0.5 Nm)
            positionGain=0.5, #0.03,
            velocityGain=1.0
        )
        
        p.stepSimulation()
        # time.sleep(1/240)

        if sim_step == (self.action_steps - 1):
            self.speed_ini = self.speed_fin
            self.steering_angle_ini = self.steering_angle_fin

    def reiniciar_estado_vehiculo(self, base_pos, base_orn_rad):
        """
        """
        # Convertir orientación a cuaternión
        quaternion = p.getQuaternionFromEuler(base_orn_rad)
        
        # Elevar el vehículo para evitar colisiones
        pos_elevada = [
            base_pos[0],
            base_pos[1],
            base_pos[2] + 0.03
        ]

        # Aplicar reinicio en posición elevada
        p.resetBasePositionAndOrientation(self.car, pos_elevada, quaternion)
        p.resetBaseVelocity(self.car, [0, 0, 0], [0, 0, 0])
        
        # Resetear articulaciones
        self.speed_ini = 0
        ang_vel = self.speed_ini / 0.0325 # v/r

        p.setJointMotorControl2(
            bodyUniqueId=self.car,
            jointIndex=self.drive_joints[0],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=ang_vel,
            force=0.03                      # Torque balanceado para 0.5kg a baja velocidad (0.02 a 0.04 Nm)
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.car,
            jointIndex=self.drive_joints[1],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-ang_vel,
            force=0.03                      # Torque balanceado para 0.5kg a baja velocidad (0.02 a 0.04 Nm)
        )

        self.steering_angle_ini = 0
        steering_angle = self.steering_angle_ini

        p.setJointMotorControl2(
            bodyUniqueId=self.car,
            jointIndex=self.steering_joints[0],
            controlMode=p.POSITION_CONTROL,
            targetPosition=steering_angle,
            force=0.4,                          # Suficiente para levantar la esquina trasera para 0.5kg (0.3 a 0.5 Nm)
            positionGain=0.5, #0.03,
            velocityGain=1.0
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.car,
            jointIndex=self.steering_joints[1],
            controlMode=p.POSITION_CONTROL,
            targetPosition=-steering_angle,
            force=0.4,                          # Suficiente para levantar la esquina trasera para 0.5kg (0.3 a 0.5 Nm)
            positionGain=0.5, #0.03,
            velocityGain=1.0
        )

        # Dar pasos de simulación para estabilizar en el aire
        # action = np.array([0.0, 0.0])
        # for _ in range(self.action_steps):
        for _ in range(20):
            # self.apply_action(action)
            p.stepSimulation()
            # time.sleep(1)
