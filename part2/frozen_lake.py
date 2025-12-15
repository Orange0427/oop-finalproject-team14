import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

# --- 創意實作：成功路徑記憶體 (PathMemory) ---
class PathMemory:
    """
    追蹤並儲存所有成功的 State-Action 序列。
    """
    def __init__(self, capacity=200): # 稍微加大記憶體
        self.success_paths = []
        self.current_trajectory = [] 
        self.capacity = capacity

    def record_step(self, state, action, reward, next_state):
        self.current_trajectory.append((state, action, reward, next_state))

    def finalize_path(self, is_success):
        if is_success:
            if len(self.success_paths) >= self.capacity:
                self.success_paths.pop(0) 
            self.success_paths.append(list(self.current_trajectory))
        self.current_trajectory = []

    def sample_batch(self, batch_size=32):
        if not self.success_paths:
            return None
        path_index = random.randrange(len(self.success_paths))
        path = self.success_paths[path_index]
        batch = random.choices(path, k=min(batch_size, len(path)))
        return batch

# -------------------------------------------------------------------------
# Part 2 核心函數
# -------------------------------------------------------------------------

def print_success_rate(rewards_per_episode):
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False, min_exp_rate=0.001):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
    
    memory = PathMemory(capacity=100) 
    REPLAY_BATCH_SIZE = 32 # 稍微降低 batch size 避免過度擬合舊路徑

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n)) 
    else:
        try:
            with open('frozen_lake8x8.pkl', 'rb') as f:
                q = pickle.load(f)
        except FileNotFoundError:
            print("評估錯誤: 找不到 frozen_lake8x8.pkl 檔案")
            return 0.0

    # === 關鍵修改 1: 學習率調低，折扣因子稍微降低 ===
    learning_rate_a = 0.087  # 低學習率適合隨機環境 (Stochastic Environment)
    discount_factor_g = 0.99 # 保持高瞻遠矚
    
    epsilon = 1 
    
    # === 關鍵修改 2: 動態計算衰減率 ===
    # 確保 epsilon 在訓練的 80% 階段才降到最低，給予充足探索時間
    epsilon_decay_rate = 1 / (episodes * 0.8) 
    
    rng = np.random.default_rng() 
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False 
        truncated = False 

        while(not terminated and not truncated):
            
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() 
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)
            
            memory.record_step(state, action, reward, new_state)

            if is_training:
                # 標準 Q-Learning 更新
                max_q_next = np.max(q[new_state,:])
                q[state,action] += learning_rate_a * (
                    reward + discount_factor_g * max_q_next - q[state,action]
                )

            state = new_state
        
        # --- Episode 結束 ---
        is_success = (reward == 1.0)
        memory.finalize_path(is_success)

        # --- 成功經驗回放 (Success Experience Replay) ---
        # 注意：我們保持較低的學習率進行回放，避免「倖存者偏差」過重
        if is_training and memory.success_paths and is_success: 
            # 修改策略：只有在「剛好成功」或「每隔幾次」時才重放，或者保持每次重放
            # 這裡保持每次重放，但依賴低學習率來穩定
            batch = memory.sample_batch(REPLAY_BATCH_SIZE)
            if batch:
                for s, a, r, ns in batch:
                    max_q_next = np.max(q[ns,:])
                    q[s,a] += learning_rate_a * (
                        r + discount_factor_g * max_q_next - q[s,a]
                    )
        
        # 衰減 epsilon
        epsilon = max(epsilon - epsilon_decay_rate, min_exp_rate)

        # === 關鍵修改 3: 移除學習率強制歸零的邏輯 ===
        # 讓 agent 在 epsilon 很低時繼續微調 Q-table

        if reward == 1:
            rewards_per_episode[i] = 1
        
        # 每 1000 次印出進度，避免看起來像當機
        if (i+1) % 5000 == 0 and is_training:
             print(f"Episode {i+1}: 目前 Epsilon {epsilon:.4f}")

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    
    if is_training == False:
        print_success_rate(rewards_per_episode)

    if is_training:
        with open("frozen_lake8x8.pkl","wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    
    # 設置 min_exploration_rate
    MIN_RATE = 0.001 

    print(f"--- 🚀 Frozen Lake (高準確度優化版) 運行 ---")
    print(f"核心策略: 低學習率 (0.1) + 長時間訓練 + 經驗回放")
    print("-" * 35)
    
    # === 關鍵修改 4: 增加訓練次數 ===
    # 8x8 Slippery 非常難收斂，建議至少 25000 ~ 30000 次
    print("開始訓練 (15,000 episodes)...")
    run(15000, is_training=True, render=False, min_exp_rate=MIN_RATE)

    print("\n開始評估 (1,000 episodes)...")
    run(1000, is_training=False, render=False, min_exp_rate=MIN_RATE)
