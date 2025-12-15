import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

def print_success_rate(rewards_per_episode):
    """計算並印出成功率"""
    total_episodes = len(rewards_per_episode)
    # 計算成功次數 (Reward = 1 代表成功到達終點)
    success_count = np.sum(rewards_per_episode == 1) 
    
    success_rate = (success_count / total_episodes) * 100
    print(f":white_check_mark: Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False):
    
    # 初始化環境：8x8 地圖，is_slippery=True (會滑，增加難度)
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    # 1. 初始化 Q-Table
    if(is_training):
        # 訓練模式：建立一個全新的 Q-Table (全 0)
        q = np.zeros((env.observation_space.n, env.action_space.n)) 
    else:
        # 測試模式：讀取已訓練好的 .pkl 檔案
        try:
            f = open('frozen_lake8x8.pkl', 'rb')
            q = pickle.load(f)
            f.close()
        except FileNotFoundError:
            print("尚未訓練，找不到 .pkl 檔案！")
            return

    # 2. 超參數設定
    learning_rate_a = 0.085 # 學習率 (Alpha)：更新 Q 值的步伐大小
    discount_factor_g = 0.99 # 折扣因子 (Gamma)：重視未來獎勵的程度
    
    epsilon = 1 # 探索率：初始為 100% 隨機探索
    
    # 衰減率：隨著回合數增加，減少隨機探索的機率
    # 這裡設定讓前 90% 的回合都有機會探索
    epsilon_decay_rate = 1 / (episodes * 0.9) 
    min_exploration_rate = 0.01 # 最小探索率保底

    rng = np.random.default_rng() 
    rewards_per_episode = np.zeros(episodes) # 記錄每回合結果

    # 3. 訓練/測試迴圈
    for i in range(episodes):
        state = env.reset()[0] # 重置環境
        terminated = False # 是否到達終點/掉洞
        truncated = False # 是否超時

        while(not terminated and not truncated):
            # 訓練模式才需要探索 (Epsilon-Greedy 策略)
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # 隨機探索
            else:
                action = np.argmax(q[state,:]) # 根據 Q 表選擇最佳動作

            new_state, reward, terminated, truncated, _ = env.step(action)

            # --- Reward Shaping (獎勵塑形) ---
            # 修改原生獎勵機制以加速收斂
            if terminated and reward == 0: 
                reward = -1  # 懲罰：掉進洞裡
            elif not terminated:
                reward = -0.01 # 懲罰：每走一步扣分 (鼓勵走捷徑)
            elif terminated and reward == 1:
                reward = 1   # 獎勵：到達終點

            # Q-Table 
            # 公式：Q_new = Q_old + alpha * (Reward + gamma * max(Q_next) - Q_old)
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        # Epsilon 衰減 (每回合結束後減少探索率)
        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)

        # 記錄成功 (這裡只記錄是否有拿到原始環境的 1 分)
        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    # 4. 繪圖與結算
    if is_training:
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            # 計算滑動平均 (看最近 100 回合的表現)
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.title(f"Training Progress (Decay: {epsilon_decay_rate:.5f})")
        plt.savefig('frozen_lake8x8.png')
        plt.close() 
        
        # 儲存模型
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()
        print(f"訓練完成。Epsilon 衰減率: {epsilon_decay_rate:.5f}")

    else:
        # 印成功率
        print(f"測試結果 (Epsilon Decay 用不到):")
        print_success_rate(rewards_per_episode)

if __name__ == '__main__':
    
    # 1. 先訓練 (不渲染畫面)
    print("--- 開始訓練 ---")
    run(15000, is_training=True, render=False) 
    
    # 2. 再測試 (顯示畫面)
    print("\n--- 開始測試 ---")
    run(1000, is_training=False, render=False)