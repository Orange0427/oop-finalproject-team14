import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def print_success_rate(rewards_per_episode):
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        try:
            f = open('frozen_lake8x8.pkl', 'rb')
            q = pickle.load(f)
            f.close()
        except FileNotFoundError:
            print("找不到模型檔案，請先訓練！")
            return

    # --- 關鍵修正參數 ---
    
    # 1. 學習率固定 0.1
    # 這是對抗隨機性(Slippery)最穩的數值，不要動它
    learning_rate_a = 0.1  
    
    # 2. 折扣因子調到極高 (0.999)
    # 原因：8x8 的安全路徑很長(可能30步+)。如果用 0.99，30步後的獎勵衰減太快，
    # 導致 AI 寧願選一條"很短但很容易死"的路(50%勝率)。
    # 改成 0.999 可以讓它願意走遠路。
    discount_factor_g = 0.999 
    
    epsilon = 1 
    
    # 3. 改用指數衰減 (Exponential Decay)
    # 線性衰減在後期會降太快，指數衰減可以讓它在中間階段多探索一陣子
    # 15000 回合，0.9996 的衰減率會在第 15000 回合降到約 0.002
    epsilon_decay_rate = 0.9996 
    
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
                # 解決 Q 值相同時的死鎖問題
                if np.all(q[state, :] == q[state, 0]):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # --- Reward Strategy ---
            # 這裡移除掉洞的 -1 懲罰。
            # 在 15000 回合的限制下，負分會讓 AI 害怕探索長路徑。
            # 我們只專注於：到達終點=1，其他=0。
            # 配合 Gamma=0.999，它會自動找到最短的"安全"路徑。
            
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        # 指數衰減
        if is_training:
            epsilon = max(epsilon * epsilon_decay_rate, 0.01)

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    if is_training:
        sum_rewards = np.zeros(episodes)
        window_size = 500
        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-window_size):(t+1)])
            sum_rewards[t] = (sum_rewards[t] / min(t+1, window_size)) * 100
            
        plt.plot(sum_rewards)
        plt.title('Success Rate (Gamma 0.999)')
        plt.xlabel('Episodes')
        plt.ylabel('Success Rate (%)')
        plt.savefig('frozen_lake8x8.png')
    
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()
        
        last_1000_success = np.sum(rewards_per_episode[-1000:]) / 1000
        print(f"最後 1000 回合平均成功率: {last_1000_success * 100:.2f}%")

    if not is_training:
        print(print_success_rate(rewards_per_episode))

if __name__ == '__main__':
    print("--- 開始訓練 ---")
    run(15000, is_training=True, render=False)
    
    print("\n--- 開始測試 ---")
    run(1000, is_training=False, render=False)