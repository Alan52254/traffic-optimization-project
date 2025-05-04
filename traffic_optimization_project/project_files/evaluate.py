# evaluate.py
import os
import traci
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ===== 基本設定 =====
SUMO_BINARY = "sumo-gui"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CONFIG = os.path.join(BASE_DIR, "osm.sumocfg")
Q_TABLE_PATH = os.path.join(BASE_DIR, "q_table.npy")
PLOT_PATH = os.path.join(BASE_DIR, "eval_reward_plot.png")

TL_ID = "joinedS_965531266_965531287_cluster_5066556756_662312633_662315056_662315165"

LANES = [
    "244450450#0_0", "244450535#0_0", "263225305#3_0", "264399255#1_0",
    "288878925#0_0", "455330015#0_0", "4860359#1_0", "501772320#1_0",
    "501772321_0", "519643766#0_0", "668070773_0", "668070789_0"
]
ACTIONS = [0, 2]
bins = [0, 10, 20]
penalty_switch = 5

# ===== 狀態與獎勵定義 =====
def discretize(state):
    return tuple(np.digitize(v, bins) for v in state)

def get_state():
    raw_state = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
    return discretize(raw_state)

def get_reward(prev_action):
    waiting_time = sum([traci.lane.getWaitingTime(lane) for lane in LANES])
    switch_penalty = penalty_switch if traci.trafficlight.getPhase(TL_ID) != prev_action else 0
    return -waiting_time - switch_penalty

# ===== 評估流程 =====
def evaluate_q_learning(max_steps=5000):
    if not os.path.exists(Q_TABLE_PATH):
        raise FileNotFoundError(f"找不到 Q-table 檔案：{Q_TABLE_PATH}")

    q_table = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    q_table.update(np.load(Q_TABLE_PATH, allow_pickle=True).item())

    try:
        traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])
    except Exception as e:
        print("無法啟動 SUMO 模擬，請確認 sumo-gui 路徑與設定檔是否正確。")
        raise e

    traci.trafficlight.setPhase(TL_ID, 0)
    total_reward = 0
    reward_list = []
    current_action = 0

    for step in range(max_steps):
        state = get_state()
        action = max(q_table[state], key=q_table[state].get, default=0)
        traci.trafficlight.setPhase(TL_ID, action)
        for _ in range(10):
            traci.simulationStep()

        reward = get_reward(current_action)
        total_reward += reward
        reward_list.append(reward)
        current_action = action

        if traci.simulation.getMinExpectedNumber() <= 0:
            break

    traci.close()

    print(f"\n 總評估 Reward：{total_reward:.2f}")
    plt.plot(reward_list)
    plt.title("Reward per Step during Evaluation")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    plt.show()

# ===== 主程式 =====
if __name__ == "__main__":
    evaluate_q_learning()
