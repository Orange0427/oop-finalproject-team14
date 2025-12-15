import gymnasium as gym
from gymnasium.envs.classic_control import MountainCarEnv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from abc import ABC, abstractmethod
import math
import pygame

# ==============================================================================
#                               PART A: 物理世界 (WORLD)
# ==============================================================================

class BaseVehicle(ABC):
    def __init__(self, name, mass, engine_power_factor, max_speed, color):
        self.name = name
        self.mass = mass
        self.engine_power_factor = engine_power_factor
        self._max_speed = max_speed
        self.color = color 

    def get_max_speed(self):
        return self._max_speed

class SportCar(BaseVehicle):
    def __init__(self):
        super().__init__(name="SportCar", mass=0.8, engine_power_factor=1.3, max_speed=0.09, color=(255, 60, 60))

class Truck(BaseVehicle):
    def __init__(self):
        super().__init__(name="Truck", mass=2.5, engine_power_factor=0.7, max_speed=0.05, color=(70, 100, 220))

class TerrainEffect(ABC):
    def __init__(self, start_pos, end_pos, name="Terrain", color=(200, 200, 200)):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.name = name
        self.color = color 

    def is_in_zone(self, position):
        return self.start_pos <= position <= self.end_pos

    @abstractmethod
    def apply_effect(self, velocity):
        pass

class BoostZone(TerrainEffect):
    def __init__(self, start_pos, end_pos, factor=1.2):
        super().__init__(start_pos, end_pos, name="Boost Zone", color=(255, 255, 100)) 
        self.factor = factor

    def apply_effect(self, velocity):
        return velocity * self.factor

class MudZone(TerrainEffect):
    def __init__(self, start_pos, end_pos, factor=0.6):
        super().__init__(start_pos, end_pos, name="Mud Zone", color=(139, 69, 19)) 
        self.factor = factor

    def apply_effect(self, velocity):
        return velocity * self.factor

class CustomMountainCarEnv(MountainCarEnv):
    def __init__(self, render_mode=None, terrain_effects=None, vehicle=None):
        super().__init__(render_mode=render_mode)
        if vehicle:
            self.vehicle = vehicle
        else:
            self.vehicle = BaseVehicle("Standard", 1.0, 1.0, 0.07, color=(50, 200, 50))
        self.terrain_effects = terrain_effects if terrain_effects else []
        self.max_speed = self.vehicle.get_max_speed()
        self.force_mag = 0.001 
        self.max_episode_steps = 200 
        self.elapsed_steps = 0      
        self.screen_width = 1000
        self.screen_height = 600

    def reset(self, *, seed=None, options=None):
        self.elapsed_steps = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        position, velocity = self.state
        force = (action - 1) * self.force_mag * self.vehicle.engine_power_factor
        gravity_effect = np.cos(3 * position) * (-0.0025) / self.vehicle.mass
        velocity += force + gravity_effect
        
        for terrain in self.terrain_effects:
            if terrain.is_in_zone(position):
                velocity = terrain.apply_effect(velocity)
        
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        if (position == self.min_position and velocity < 0): velocity = 0 
        terminated = bool(position >= self.goal_position)
        
        # ==========================================
        # 🔧 獎勵塑形 (Reward Shaping) - 高度獎勵版
        # ==========================================
        
        # 1. 基礎懲罰：每一秒都扣分，逼它快點跑
        reward = -1.0  

        # 2. 🔥 新增：高度獎勵 (越接近旗子分數越高)
        # 谷底大約是 -0.5，終點是 0.5
        # 當位置高於谷底時，給予一點點甜頭，但總分還是負的 (避免 AI 刷分)
        if position > -0.5:
            reward += (position + 0.5) 

        # 3. 過關大獎：這一筆最大，確保過關才是唯一目標
        if terminated:
            reward += 1000.0  
            
        self.state = (position, velocity)
        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None: return
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None: self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale_x = self.screen_width / world_width
        scale_y = 450 
        offset_y = 150 

        def get_screen_y(world_y_norm):
            return (1 - world_y_norm) * scale_y + offset_y

        self.screen.fill((255, 255, 255))
        
        for terrain in self.terrain_effects:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            points = []
            for px in range(int((terrain.start_pos - self.min_position) * scale_x), 
                            int((terrain.end_pos - self.min_position) * scale_x), 2):
                world_x = (px / scale_x) + self.min_position
                world_y_norm = np.sin(3 * world_x) * 0.45 + 0.55
                py = get_screen_y(world_y_norm)
                points.append((px, py))
            
            if points:
                points.append((points[-1][0], self.screen_height)) 
                points.append((points[0][0], self.screen_height)) 
                color_with_alpha = terrain.color + (128,) 
                pygame.draw.polygon(overlay, color_with_alpha, points)
                font = pygame.font.SysFont("Arial", 20, bold=True)
                text = font.render(terrain.name, True, (80, 80, 80))
                overlay.blit(text, (points[0][0] + 10, points[0][1] + 30))
            self.screen.blit(overlay, (0, 0))

        track_points = []
        for x in np.linspace(self.min_position, self.max_position, 200):
            world_y_norm = np.sin(3 * x) * 0.45 + 0.55
            px = (x - self.min_position) * scale_x
            py = get_screen_y(world_y_norm)
            track_points.append((px, py))
        pygame.draw.aalines(self.screen, (0, 0, 0), False, track_points) 

        pos = self.state[0]
        car_x = (pos - self.min_position) * scale_x
        car_y_norm = np.sin(3 * pos) * 0.45 + 0.55
        car_y = get_screen_y(car_y_norm)

        next_pos = pos + 0.01
        next_car_x = (next_pos - self.min_position) * scale_x
        next_car_y_norm = np.sin(3 * next_pos) * 0.45 + 0.55
        next_car_y = get_screen_y(next_car_y_norm)

        dx = next_car_x - car_x
        dy = next_car_y - car_y
        angle = math.degrees(math.atan2(-dy, dx))

        if self.vehicle.name == "Truck":
            w, h = 66, 36 
            offset_car_y = 18 
            car_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(car_surf, self.vehicle.color, (0, 0, 50, 30), border_radius=4)
            pygame.draw.rect(car_surf, (150, 150, 150), (50, 6, 16, 24), border_radius=3)
            pygame.draw.circle(car_surf, (0,0,0), (15, 30), 6)
            pygame.draw.circle(car_surf, (0,0,0), (56, 30), 6)
        elif self.vehicle.name == "SportCar":
            w, h = 54, 24 
            offset_car_y = 12
            car_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.ellipse(car_surf, self.vehicle.color, (0, 0, 54, 18))
            pygame.draw.polygon(car_surf, (0,0,0), [(0, 0), (8, -6), (16, 0)])
            pygame.draw.circle(car_surf, (30,30,30), (12, 18), 6)
            pygame.draw.circle(car_surf, (30,30,30), (42, 18), 6)
            pygame.draw.rect(car_surf, (200, 255, 255), (26, 3, 12, 8))
        else:
            w, h = 45, 24 
            offset_car_y = 12
            car_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(car_surf, self.vehicle.color, (0, 0, 45, 18), border_radius=5)
            pygame.draw.circle(car_surf, (0,0,0), (9, 18), 6)
            pygame.draw.circle(car_surf, (0,0,0), (36, 18), 6)

        rotated_car = pygame.transform.rotate(car_surf, angle)
        rect = rotated_car.get_rect(center=(car_x, car_y - offset_car_y))
        self.screen.blit(rotated_car, rect)
        
        goal_x = (self.goal_position - self.min_position) * scale_x
        goal_y_norm = np.sin(3 * self.goal_position) * 0.45 + 0.55
        goal_y = get_screen_y(goal_y_norm)
        
        pygame.draw.line(self.screen, (50, 50, 50), (goal_x, goal_y), (goal_x, goal_y - 80), 5)
        pygame.draw.polygon(self.screen, (255, 0, 0), 
                           [(goal_x, goal_y - 80), (goal_x + 50, goal_y - 60), (goal_x, goal_y - 40)])
        pygame.draw.polygon(self.screen, (0, 0, 0), 
                           [(goal_x, goal_y - 80), (goal_x + 50, goal_y - 60), (goal_x, goal_y - 40)], 2)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)) if self.render_mode == "rgb_array" else None


# ==============================================================================
#                               PART B: 智能體大腦 (BRAIN)
# ==============================================================================

class ExplorationStrategy(ABC):
    @abstractmethod
    def select_action(self, q_values, action_space_n): pass
    @abstractmethod
    def update(self): pass

class EpsilonGreedyStrategy(ExplorationStrategy):
    def __init__(self, epsilon=1.0, min_epsilon=0.01, decay_rate=0.0005):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.rng = np.random.default_rng()

    def select_action(self, q_values, action_space_n):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(action_space_n)
        else:
            return np.argmax(q_values)

    def update(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate)

class Agent(ABC):
    def __init__(self, name, n_actions):
        self.name = name
        self.n_actions = n_actions

    @abstractmethod
    def choose_action(self, state, is_training=True): pass
    @abstractmethod
    def learn(self, state, action, reward, next_state, terminated): pass
    def save(self, filename): pass
    def load(self, filename): pass

class QLearningAgent(Agent):
    def __init__(self, name, n_states_pos, n_states_vel, n_actions, strategy, lr=0.1, gamma=0.99):
        super().__init__(name, n_actions)
        self.q_table = np.zeros((n_states_pos, n_states_vel, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.strategy = strategy

    def choose_action(self, state, is_training=True):
        state_p, state_v = state
        q_values = self.q_table[state_p, state_v, :]
        return self.strategy.select_action(q_values, self.n_actions) if is_training else np.argmax(q_values)

    def learn(self, state, action, reward, next_state, terminated):
        sp, sv = state
        nsp, nsv = next_state
        current_q = self.q_table[sp, sv, action]
        max_next_q = np.max(self.q_table[nsp, nsv, :])
        target = reward + (0 if terminated else self.gamma * max_next_q)
        self.q_table[sp, sv, action] += self.lr * (target - current_q)
        
    def update_strategy(self):
        self.strategy.update()

    def save(self, filename):
        with open(filename, 'wb') as f: pickle.dump(self.q_table, f)
        print(f"💾 [{self.name}] Model saved -> {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                try:
                    loaded = pickle.load(f)
                    if loaded.shape == self.q_table.shape:
                        self.q_table = loaded
                        print(f"✅ [{self.name}] Loaded -> {filename}")
                        return True
                except: pass
        print(f"⚠️ [{self.name}] No valid model found ({filename}), new one created.")
        return False

class SarsaAgent(QLearningAgent):
    def learn(self, state, action, reward, next_state, terminated):
        sp, sv = state
        nsp, nsv = next_state
        current_q = self.q_table[sp, sv, action]
        q_next_values = self.q_table[nsp, nsv, :]
        next_action = self.strategy.select_action(q_next_values, self.n_actions)
        next_q = self.q_table[nsp, nsv, next_action]
        target = reward + (0 if terminated else self.gamma * next_q)
        self.q_table[sp, sv, action] += self.lr * (target - current_q)

class RuleBasedAgent(Agent):
    def __init__(self, n_actions): super().__init__("Pro (Rule-Based)", n_actions)
    def choose_action(self, state, is_training=False):
        _, v_idx = state
        return 2 if v_idx > 10 else 0
    def learn(self, *args): pass

class RandomAgent(Agent):
    def __init__(self, n_actions): super().__init__("Rookie (Random)", n_actions)
    def choose_action(self, state, is_training=False): return np.random.choice(self.n_actions)
    def learn(self, *args): pass


# ==============================================================================
#                               PART C: 實驗室管理 (MANAGER)
# ==============================================================================

class MountainCarWrapper:
    def __init__(self, vehicle_type='Standard', agent_name='unknown', render_mode=None, bins=20, record_video=False, video_folder='./videos', terrain='NoMud'):
        self.bins = bins
        
        # 🌲 地形工廠
        if terrain == 'NoMud':
            print("🌲 Terrain: NoMud (Boost Only)")
            terrain_config = [BoostZone(start_pos=-0.6, end_pos=-0.3, factor=1.2)]
        elif terrain == 'Mud':
            print("🌲 Terrain: Mud (Mud + Boost)")
            terrain_config = [
                BoostZone(start_pos=-0.6, end_pos=-0.3, factor=1.2),
                MudZone(start_pos=0.2, end_pos=0.4, factor=0.9) 
            ]
        elif terrain == 'HardMud':
            print("🌲 Terrain: HardMud (Sticky Mud!)")
            terrain_config = [
                BoostZone(start_pos=-0.6, end_pos=-0.3, factor=1.2),
                MudZone(start_pos=0.2, end_pos=0.4, factor=0.6)
            ]
        else:
            print("🌲 Terrain: Default (No effects)")
            terrain_config = []

        if vehicle_type == 'SportCar': vehicle = SportCar()
        elif vehicle_type == 'Truck': vehicle = Truck()
        else: vehicle = BaseVehicle("Standard", 1.0, 1.0, 0.07, color=(50, 200, 50))
        
        actual_render_mode = 'rgb_array' if record_video else render_mode
        raw_env = CustomMountainCarEnv(render_mode=actual_render_mode, terrain_effects=terrain_config, vehicle=vehicle)
        
        if record_video:
            # 🔥 修改點：改成 True，每一場都錄影
            trigger = lambda ep_id: True 
            folder_name = f"{video_folder}/{agent_name}_{vehicle_type}_{terrain}"
            self.env = gym.wrappers.RecordVideo(
                raw_env, 
                folder_name, 
                episode_trigger=trigger, 
                name_prefix=agent_name,
                disable_logger=True
            )
        else:
            self.env = raw_env
        
        self.pos_space = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], bins)
        self.vel_space = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], bins)

    def reset(self):
        state, _ = self.env.reset()
        return self._discretize(state)
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        return self._discretize(next_state), reward, terminated, truncated
    
    def _discretize(self, state):
        p_idx = np.digitize(state[0], self.pos_space) - 1
        v_idx = np.digitize(state[1], self.vel_space) - 1
        return (np.clip(p_idx, 0, self.bins-1), np.clip(v_idx, 0, self.bins-1))
    
    def close(self): self.env.close()
    
    @property
    def action_space_n(self): return self.env.action_space.n

class LabManager:
    def __init__(self, args):
        self.args = args
        self.bins = 40
        temp = gym.make('MountainCar-v0')
        self.n_actions = temp.action_space.n
        temp.close()

    def _create_agent(self, agent_type, vehicle_name, training=False):
        strategy = EpsilonGreedyStrategy(epsilon=1.0 if training else 0.0)
        params = {'n_states_pos': self.bins, 'n_states_vel': self.bins, 'n_actions': self.n_actions}
        filename = f"model_{agent_type}_{vehicle_name}_{self.args.terrain}.pkl"

        if agent_type == 'qlearning':
            agent = QLearningAgent(f"QLearning", **params, strategy=strategy)
            agent.load(filename)
            return agent
        elif agent_type == 'sarsa':
            agent = SarsaAgent(f"SARSA", **params, strategy=strategy)
            agent.load(filename)
            return agent
        elif agent_type == 'pro': return RuleBasedAgent(self.n_actions)
        elif agent_type == 'rookie': return RandomAgent(self.n_actions)
        else: raise ValueError("Unknown agent")

    def train(self):
        print(f"\n🏋️ TRAINING: {self.args.agent} on {self.args.vehicle} [Map: {self.args.terrain}]")
        
        env = MountainCarWrapper(
            self.args.vehicle, 
            agent_name=self.args.agent,
            bins=self.bins, 
            record_video=self.args.record, 
            terrain=self.args.terrain
        )
        agent = self._create_agent(self.args.agent, self.args.vehicle, training=True)
        
        rewards_history = []
        for i in range(self.args.episodes):
            state = env.reset()
            done = False
            total = 0
            while not done:
                action = agent.choose_action(state, is_training=True)
                next_state, reward, term, trunc = env.step(action)
                done = term or trunc
                agent.learn(state, action, reward, next_state, term)
                state = next_state
                total += reward
            
            if hasattr(agent, 'update_strategy'): agent.update_strategy()
            rewards_history.append(total)
            
            if (i+1) % 100 == 0:
                print(f"Ep {i+1}: Avg Reward={np.mean(rewards_history[-100:]):.1f}, Epsilon={agent.strategy.epsilon:.2f}")

        agent.save(f"model_{self.args.agent}_{self.args.vehicle}_{self.args.terrain}.pkl")
        env.close()
        self._plot_curve(rewards_history)

    def play(self):
        print(f"\n🎮 PLAY MODE: {self.args.agent} [Map: {self.args.terrain}]")
        
        render_mode = 'human' if not self.args.record else None
        
        env = MountainCarWrapper(
            self.args.vehicle, 
            agent_name=self.args.agent,
            render_mode=render_mode, 
            bins=self.bins, 
            record_video=self.args.record, 
            terrain=self.args.terrain
        )
        agent = self._create_agent(self.args.agent, self.args.vehicle, training=False)
        
        play_eps = 5 if self.args.record else 3
        for i in range(play_eps):
            state = env.reset()
            done = False
            total = 0
            while not done:
                action = agent.choose_action(state, is_training=False)
                state, reward, term, trunc = env.step(action)
                done = term or trunc
                total += reward
            print(f"Episode {i+1} Score: {total:.1f}")
        env.close()

    def compare(self):
        print(f"\n🏎️ COMPARISON on {self.args.vehicle} [Map: {self.args.terrain}]")
        env = MountainCarWrapper(self.args.vehicle, render_mode=None, bins=self.bins, terrain=self.args.terrain)
        
        agents = [
            self._create_agent('rookie', self.args.vehicle),
            self._create_agent('pro', self.args.vehicle),
            self._create_agent('qlearning', self.args.vehicle),
            self._create_agent('sarsa', self.args.vehicle)
        ]
        
        results = {}
        for agent in agents:
            print(f"Testing {agent.name}...", end=" ", flush=True)
            scores = []
            for _ in range(self.args.episodes):
                state = env.reset()
                done = False
                total = 0
                while not done:
                    action = agent.choose_action(state, is_training=False)
                    state, reward, term, trunc = env.step(action)
                    done = term or trunc
                    total += reward
                scores.append(total)
            results[agent.name] = scores
            print(f"Avg: {np.mean(scores):.1f}")
        
        env.close()
        self._plot_boxplot(results)

    def _plot_curve(self, rewards):
        plt.figure(figsize=(10,5))
        plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'))
        plt.title(f"Training Curve ({self.args.agent} - {self.args.terrain})")
        plt.savefig(f"train_{self.args.agent}_{self.args.vehicle}_{self.args.terrain}.png")
        plt.close()

    def _plot_boxplot(self, results):
        plt.figure(figsize=(10,6))
        plt.boxplot(results.values(), labels=results.keys(), patch_artist=True)
        plt.title(f"Comparison ({self.args.vehicle} - {self.args.terrain})")
        plt.savefig(f"compare_{self.args.vehicle}_{self.args.terrain}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mountain Car AI - Full Version")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--play', action='store_true')
    group.add_argument('--compare', action='store_true')
    
    parser.add_argument('--agent', type=str, default='qlearning', choices=['qlearning', 'sarsa', 'rookie', 'pro'])
    parser.add_argument('--vehicle', type=str, default='Standard', choices=['Standard', 'SportCar', 'Truck'])
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--record', action='store_true', help="Record video")
    
    parser.add_argument('--terrain', type=str, default='NoMud', choices=['NoMud', 'Mud', 'HardMud'], 
                        help="Choose terrain: NoMud (no mud), Mud (mild), HardMud (sticky)")

    manager = LabManager(parser.parse_args())
    
    if manager.args.train: manager.train()
    elif manager.args.play: manager.play()
    elif manager.args.compare: manager.compare()