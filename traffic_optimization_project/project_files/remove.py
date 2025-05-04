# delete_files.py
import os
import shutil

# 移除舊的 Q-table
if os.path.exists("project_files/q_table.npy"):
    os.remove("project_files/q_table.npy")
    print("[Info] q_table.npy 已刪除")

# 移除 reward 圖表
if os.path.exists("project_files/reward_plot.png"):
    os.remove("project_files/reward_plot.png")
    print("[Info] reward_plot.png 已刪除")

# 清空 training_logs 資料夾
if os.path.exists("training_logs"):
    shutil.rmtree("training_logs")
    print("[Info] training_logs 資料夾已刪除")

# 重新建立 training_logs 資料夾
os.makedirs("training_logs")
print("[Info] 已重新建立 training_logs 資料夾")
