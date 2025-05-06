# Traffic Signal Optimization using Q-Learning and SUMO

本專案針對智慧城市中的交通擁塞問題，實作一個基於 Q-Learning 的交通號誌控制系統，並透過 SUMO 進行模擬。系統能根據即時交通狀況，動態調整紅綠燈相位，以提升通行效率、降低車輛等待時間。

## 專案目標

- 使用強化學習中的 Q-Learning 方法訓練 AI Agent
- 以 SUMO 模擬台北車站路網並控制紅綠燈
- 分析訓練過程與結果，並與傳統固定時序策略比較效能

## 專案架構

traffic-optimization-project/ <br>├── training_logs/ # 儲存每回合訓練過程資料（CSV） <br> ├── training_log_episode_1.csv <br>└── ...<br> ├── traffic_log.csv # 統一車流記錄（如需額外使用）<br> ├── project_files/ # SUMO 模擬所需檔案 <br>├── osm.sumocfg <br> ├── osm.net.xml.gz <br>└── ...（包含 .rou.xml, .poly.xml.gz 等）<br> ├── q_learning.py # Q-Learning 實作主程式 <br>├── q_table.npy # 訓練完成後的 Q-table <br> ├── run.bat # 一鍵執行腳本 <br>├── build.bat # （可選）初始建構腳本 <br>└── README.md # 本說明文件

##  系統說明

- **狀態空間（State）**：12 條重要車道的車輛數量
- **動作空間（Actions）**：相位變更（phase 0 或 2）
- **獎勵函數（Reward）**：負的總等待時間（鼓勵減少壅塞）
- **學習參數**
  - α（學習率）：0.1
  - γ（折扣因子）：0.9
  - ε（探索率）：0.1
  - 訓練回合：20

## 模擬地點

本模擬場景選用 **台北車站周邊區域**，透過 SUMO 的 `osmWebWizard.py` 下載 OSM 路網，並設定車流與交通號誌控制節點。
ython q_learning.py
系統會執行 20 回合訓練，並於 training_logs/ 儲存每回合資料，結束後產出 q_table.npy。

3. 查看模擬過程
訓練過程會自動開啟 sumo-gui 介面，觀察車流與紅綠燈變化。

## 📊 成果
每回合 reward 趨勢圖顯示學習過程逐步改善壅塞。
最終可觀察到等待時間降低、決策穩定。


## 👥 成員
組別：Nine Yin Manuel
<br>組員：Alan52254, Hjc

## 📌 未來改進方向
延伸多個路口協調控制

融入排放量、多目標學習

採用 DQN、DDPG 等深度強化學習模型強化效能
