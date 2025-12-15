
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

def print_success_rate(rewards_per_episode):
    """計算並印出成功率"""
    total_episodes = len(rewards_per_episode)
    # 注意：因為你有修改 reward (變成 -1 或 -0.01)，
    # 這裡原本的 np.sum 已經不能直接代表成功次數了。
    # 我們改用 "有多少回合大於 0" 來判斷是否成功到達終點 (終點原本是 +1)
    success_count = np.sum(rewards_per_episode == 1) 
    
    success_rate = (success_count / total_episodes) * 100
    print(f":white_check_mark: Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    # 1. 初始化 Q-Table
    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) 
    else:
        try:
            f = open('frozen_lake8x8.pkl', 'rb')
            q = pickle.load(f)
            f.close()
        except FileNotFoundError:
            print("尚未訓練，找不到 .pkl 檔案！")
            return

    # 2. 超參數設定
    learning_rate_a = 0.085 # 修正：降低學習率以適應滑動環境
    discount_factor_g = 0.99 # 修正：提高遠見，因為 8x8 路徑很長
    
    epsilon = 1 
    
    # 修正：確保變數在任何模式下都有定義，避免報錯
    # 衰減率邏輯優化：我們希望他在訓練的前半段(例如前 10000 回合)都在探索
    # 數值大約在 0.0001 上下才是合理的
    epsilon_decay_rate = 1 / (episodes * 0.9) 
    min_exploration_rate = 0.01

    rng = np.random.default_rng() 
    rewards_per_episode = np.zeros(episodes)

    # 3. 訓練/測試迴圈
    for i in range(episodes):
        state = env.reset()[0] 
        terminated = False 
        truncated = False 

        while(not terminated and not truncated):
            # 訓練模式才需要探索
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() 
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # --- Reward Shaping (獎勵塑形) ---
            if terminated and reward == 0: 
                reward = -1  # 掉洞懲罰
            elif not terminated:
                reward = -0.01 # 步數懲罰 (鼓勵走快點)
            elif terminated and reward == 1:
                reward = 1   # 到達終點 (保持原本獎勵)

            # 更新 Q-Table
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        # Epsilon 衰減 (每一回合結束後做一次即可，不用在 while 裡面做)
        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)

        # 記錄成功 (這裡我們只記錄真的到達終點的情況)
        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    # 4. 繪圖與結算
    # 只有訓練時才需要畫圖
    if is_training:
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.title(f"Training Progress (Decay: {epsilon_decay_rate:.5f})")
        plt.savefig('frozen_lake8x8.png')
        plt.close() # 記得關閉圖表，不然在迴圈中會記憶體溢出
        
        # 儲存模型
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()
        print(f"訓練完成。Epsilon 衰減率: {epsilon_decay_rate:.5f}")

    else:
        # 測試模式只印成功率
        print(f"測試結果 (Epsilon Decay 用不到):")
        print_success_rate(rewards_per_episode)

if __name__ == '__main__':
    
    # 1. 先訓練 (不渲染畫面)
    print("--- 開始訓練 ---")
    run(15000, is_training=True, render=False) 
    
    # 2. 再測試 (顯示畫面)
    print("\n--- 開始測試 ---")
    run(1000, is_training=False, render=False)
