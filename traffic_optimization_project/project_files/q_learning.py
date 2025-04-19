import os
import traci
import numpy as np
import random
import csv
from collections import defaultdict

SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "osm.sumocfg"
TL_ID = "joinedS_965531266_965531287_cluster_5066556756_662312633_662315056_662315165"

# 可觀測的車道（從網路中挑選重要進出口）
LANES = [
    "244450450#0_0", "244450535#0_0", "263225305#3_0", "264399255#1_0",
    "288878925#0_0", "455330015#0_0", "4860359#1_0", "501772320#1_0",
    "501772321_0", "519643766#0_0", "668070773_0", "668070789_0"
]

# Q-learning 參數
alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 20
ACTIONS = [0, 2]

def get_state():
    return tuple([traci.lane.getLastStepVehicleNumber(lane) for lane in LANES])

def get_reward():
    total_waiting = sum([traci.lane.getWaitingTime(lane) for lane in LANES])
    return -total_waiting

def run_episode(q_table, episode_id):
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
    step = 0
    traci.trafficlight.setPhase(TL_ID, 0)

    # 開啟 CSV 檔案寫入模式
    csv_filename = f"training_log_episode_{episode_id+1}.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        fieldnames = ["episode", "step", "state", "action", "reward", "phase"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while traci.simulation.getMinExpectedNumber() > 0 and step < 5000:
            state = get_state()

            # epsilon-greedy 決策
            if random.uniform(0, 1) < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(q_table[state], key=q_table[state].get, default=0)

            # 套用相位
            traci.trafficlight.setPhase(TL_ID, action)
            for _ in range(10):
                traci.simulationStep()

            # 計算更新
            next_state = get_state()
            reward = get_reward()
            best_future_q = max(q_table[next_state].values(), default=0)
            q_table[state][action] += alpha * (reward + gamma * best_future_q - q_table[state][action])

            # 記錄資料
            writer.writerow({
                "episode": episode_id + 1,
                "step": step,
                "state": str(state),
                "action": action,
                "reward": reward,
                "phase": action
            })

            step += 1

    traci.close()

def main():
    q_table = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for episode in range(num_episodes):
        print(f"Training episode {episode + 1}/{num_episodes}")
        run_episode(q_table, episode)

    print("Training complete. Saving Q-table.")
    np.save("q_table.npy", dict(q_table))

if __name__ == "__main__":
    main()
