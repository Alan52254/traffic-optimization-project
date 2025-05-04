# train.py (增強版：加入切燈懲罰、狀態離散化 bins、改進 reward function)

import os
import traci
import numpy as np
import random
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

SUMO_BINARY = "sumo"  # 本地建議使用 headless 模式
SUMO_CONFIG = "project_files/osm.sumocfg"
TL_ID = "joinedS_965531266_965531287_cluster_5066556756_662312633_662315056_662315165"

LANES = [
    "244450450#0_0", "244450535#0_0", "263225305#3_0", "264399255#1_0",
    "288878925#0_0", "455330015#0_0", "4860359#1_0", "501772320#1_0",
    "501772321_0", "519643766#0_0", "668070773_0", "668070789_0"
]

# Q-Learning 參數設定
alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 1000
ACTIONS = [0, 2]  # 綠燈方向（0: 東西向，2: 南北向）
bins = [0, 10, 20]  # 狀態離散化
penalty_switch = 5  # 切換紅綠燈的懲罰值

# 離散化狀態空間
def discretize(state):
    return tuple(np.digitize(v, bins) for v in state)

def get_state():
    raw_state = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
    return discretize(raw_state)

def get_reward(prev_action):
    waiting_time = sum([traci.lane.getWaitingTime(lane) for lane in LANES])
    switch_penalty = penalty_switch if traci.trafficlight.getPhase(TL_ID) != prev_action else 0
    return -waiting_time - switch_penalty

def run_episode(q_table, episode_id, reward_log):
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
    step = 0
    current_action = 0
    traci.trafficlight.setPhase(TL_ID, current_action)

    csv_filename = f"training_logs/training_log_episode_{episode_id+1}.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        fieldnames = ["episode", "step", "state", "action", "reward", "phase"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_reward = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < 5000:
            state = get_state()

            if random.uniform(0, 1) < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(q_table[state], key=q_table[state].get, default=0)

            traci.trafficlight.setPhase(TL_ID, action)
            for _ in range(10):
                traci.simulationStep()

            next_state = get_state()
            reward = get_reward(current_action)
            total_reward += reward

            best_future_q = max(q_table[next_state].values(), default=0)
            q_table[state][action] += alpha * (reward + gamma * best_future_q - q_table[state][action])

            writer.writerow({
                "episode": episode_id + 1,
                "step": step,
                "state": str(state),
                "action": action,
                "reward": reward,
                "phase": action
            })

            current_action = action
            step += 1

    reward_log.append(total_reward)
    traci.close()

def main():
    os.makedirs("training_logs", exist_ok=True)
    q_table = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    reward_log = []

    for episode in range(num_episodes):
        print(f"Training episode {episode + 1}/{num_episodes}")
        run_episode(q_table, episode, reward_log)

    print("Training complete. Saving Q-table.")
    np.save("project_files/q_table.npy", dict(q_table))

    plt.plot(range(1, num_episodes + 1), reward_log, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward per Episode")
    plt.grid(True)
    plt.savefig("project_files/reward_plot.png")
    plt.show()

if __name__ == "__main__":
    main()


