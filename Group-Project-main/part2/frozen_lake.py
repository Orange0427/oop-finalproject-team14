import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    # --- 1. 環境設定 ---
    # render_mode: 訓練時不渲染以加快速度，測試時可開啟
    render_mode = 'human' if render else None
    
    # map_name="8x8", is_slippery=True 是最難的設定
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode=render_mode)

    # --- 2. Q-Table 初始化 ---
    if(is_training):
        # 採用 Code 1 的技巧：用極小的隨機數初始化，避免全 0 的死板，有助於打破對稱性
        q = np.random.uniform(low=0, high=0.01, size=(env.observation_space.n, env.action_space.n))
    else:
        # 測試模式：讀取存好的模型
        try:
            f = open('frozen_lake8x8.pkl', 'rb')
            q = pickle.load(f)
            f.close()
        except FileNotFoundError:
            print("錯誤：找不到模型檔案 'frozen_lake8x8.pkl'，請先執行訓練模式。")
            return np.zeros(episodes)

    # --- 3. 參數設定 ---
    # 基礎參數
    discount_factor_g = 0.99  # 8x8 路徑長，需要看得遠
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    
    # 變數初始化
    epsilon = 1.0 
    learning_rate_a = 0.1 # 初始值，稍後會動態調整

    # --- 4. 主迴圈 ---
    for i in range(episodes):
        
        # [策略融合] 採用 Code 1 的三階段動態參數控制
        # 這比單純的線性衰減更能適應 8x8 的困難度
        if is_training:
            if i < int(episodes * 0.5): # 前 50%：探索期
                learning_rate_a = 0.2 #學習率高
                epsilon_decay = 1 / (episodes * 0.6) #緩慢下降
                epsilon = max(0.1, epsilon - epsilon_decay) # 讓 epsilon 緩慢下降(強迫Agent探索未知的領域)
            elif i < int(episodes * 0.9): # 中間 40%：收斂期
                learning_rate_a = 0.05
                epsilon_decay = 1 / (episodes * 0.2) # 加速收斂
                epsilon = max(0.01, epsilon - epsilon_decay)
            else: # 最後 10%：衝刺/微調期
                learning_rate_a = 0.01
                epsilon = 0.0 # 純利用已知知識
        
        state = env.reset()[0]
        terminated = False      
        truncated = False       
        
        # 紀錄原始獎勵用來計算成功率 (不含懲罰扣分)
        success = False

        while(not terminated and not truncated):
            # [動作選擇]
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() 
            else:
                # Tie-breaking: 加入微小雜訊避免 Argmax 總是選同一個索引導致卡死
                action = np.argmax(q[state, :] + rng.random(env.action_space.n) * 1e-9)

            new_state, reward, terminated, truncated, _ = env.step(action)

            # [策略融合] 採用 Code 2 的 Reward Shaping (獎勵塑形)
            # 這對 8x8 非常重要，否則 Agent 很難在隨機漫步中摸到終點
            original_reward = reward # 暫存原始獎勵判斷成功與否
            
            if is_training:
                if terminated and reward == 0:
                    reward = -1     # 掉洞懲罰
                elif not terminated:
                    reward = -0.01  # 步數懲罰 (鼓勵走快點)
                elif terminated and reward == 1:
                    reward = 1      # 到達終點

                # Q-Learning 更新公式
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state
            
            # 判斷是否真的成功 (原始 reward 為 1)
            if original_reward == 1:
                success = True

        # 紀錄本回合結果
        if success:
            rewards_per_episode[i] = 1
            
        # [監控進度] 每 1000 回合印一次資訊
        if is_training and (i+1) % 1000 == 0:
            recent_acc = np.mean(rewards_per_episode[i-100:i]) * 100
            print(f"Episode {i+1}: Epsilon {epsilon:.4f}, LR {learning_rate_a:.3f}, Last 100 Success Rate: {recent_acc:.1f}%")

    env.close()

    # --- 5. 繪圖與存檔 (僅訓練模式) ---
    if is_training:
        # 計算移動平均成功率
        sum_rewards = np.zeros(episodes)
        window = 500 
        for t in range(episodes):
            start_idx = max(0, t - window)
            # 避免除以 0
            div = max(1, t - start_idx + 1)
            sum_rewards[t] = np.sum(rewards_per_episode[start_idx:(t+1)]) / div * 100
        
        plt.figure(figsize=(10, 5))
        plt.plot(sum_rewards)
        plt.title('Training Success Rate (Moving Average)')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (%)')
        plt.grid(True)
        plt.savefig('frozen_lake8x8_combined.png')
        print(f"\n圖表已儲存為 frozen_lake8x8_combined.png")
        
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()
        print("Model Saved!")

    return rewards_per_episode

def evaluate(episodes):
    print(f"\n--- Starting Evaluation for {episodes} episodes ---")
    # 測試時不訓練 (is_training=False)，若要看畫面可改 render=True
    rewards = run(episodes, is_training=False, render=False)
    
    success_count = np.sum(rewards)
    success_rate = success_count / episodes * 100
    
    print(f"Final Evaluation Success Rate: {success_rate:.2f}% ({int(success_count)} / {episodes} wins)")
    return success_rate

if __name__ == '__main__':
    # 1. 訓練階段 (建議至少 15000 回合，因為 8x8 且 slippery 很難)
    print("--- Training Start ---")
    run(15000, is_training=True, render=False)
    
    # 2. 測試階段
    evaluate(1000)
    
    # 3. 若想看實際跑的樣子，可以解開下面這行 (只跑 5 次)
    # run(5, is_training=False, render=True)