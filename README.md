# 🧠 Invisible Spectrum Environment (ISE)

<div align="center">
  <h3>A Subtle Cognitive Pattern Recognition Environment for OpenEnv RL Agents</h3>
</div>

---

## 📌 1. The Core Problem

Traditional neurodevelopmental evaluation systems are inherently biased towards overt, visible behavioral manifestations. This creates a significant real-world gap: individuals who have learned to **"mask"** or compensate for neurodevelopmental traits (like ADHD/ASD) often go entirely undetected by surface-level heuristics. 

Traditional heuristic bots fail because they look at static snapshots. **Invisible Spectrum Environment (ISE)** is a dynamic, continuous-observation Reinforcement Learning environment built for the **OpenEnv Hackathon**. It trains RL agents to identify underlying structural patterns by analyzing *how behavior degrades under sequential cognitive stress over time*.

> **Disclaimer:** This environment assigns a probabilistic cluster classification based on behavioral structures for AI research purposes. It is **not** a medical diagnostic tool.

---

## ⚙️ 2. Environment Architecture & OpenEnv Spec

The environment seamlessly integrates with the OpenEnv specification, interacting with the agent using structured `Pydantic` schemas representing real-world API expectations.

### 📊 Observation Space
Agents receive continuous signal feedback `[0.0, 1.0]` at each step:
*   `response_time`: Simulated speed of reaction (hesitation under distress).
*   `attention`: Granular focus stability. 
*   `consistency`: Step-to-step behavioral variance.
*   `difficulty`: Contextual difficulty matrix parameterization.

### 🎮 Action Space
The agent can query dynamically and classify when confident:
*   `{"action_type": "ask_easy"}` - Simple behavioral probing.
*   `{"action_type": "ask_hard"}` - Complex structural probing (causes masked traits to slip).
*   `{"action_type": "classify", "value": "normal" | "adhd" | "masked"}` - Final predictive clustering.

### 🧮 Reward Logic
*   **Information Gathering (`ask_x`)**: `-0.02` Step Penalty (encourages efficiency and prevents infinite looping).
*   **Correct Classification (`classify`)**: `+1.0` Reward.
*   **Wrong Classification**: `-1.0` Penalty.
*   **Timeout (Max steps)**: `-0.5` Penalty.

---

## 👤 3. Task Levels & Masking Mechanics

Our signature feature represents the underrepresented **Masked Profile**.

1.  **Task 1 (Easy)**: High differentiation between normal and ADHD profiles. Minimal noise.
2.  **Task 2 (Medium)**: Added Gaussian overlaps. Normal and ADHD parameters blend, requiring deeper sequential probing.
3.  **Task 3 (Hard)**: Focuses heavily on the **Masked** profile. Masked profiles will perfectly mimic a "Normal" profile initially. However, after `steps > 5` or when confronted with `ask_hard` actions, their attention and consistency dynamically degenerate, requiring temporal RL logic to detect.

---

## 🚀 4. Setup & Deployment Instructions

The environment is container-ready, equipped with `inference.py` to test locally or via Hugging Face.

**Local Run without Docker:**
```bash
pip install -r requirements.txt
python inference.py
```

**Docker Run (Evaluation Ready):**
```bash
docker build -t ise-env .
docker run --rm ise-env
```

---

## 🏆 5. Validation Results

We include an adaptive heuristic `inference.py` baseline to demonstrate functionality.

| Modality        | Target Validation    | Simulated Heuristic Performance | 
|-----------------|----------------------|---------------------------------|
| **Easy Task**   | `0.75 - 0.85`        | `~0.80`                         |
| **Medium Task** | `0.55 - 0.65`        | `~0.59`                         | 
| **Hard Task**   | `0.45 - 0.60`        | `~0.45`                         |

**Interpretation:** The deterministic grader rewards efficiency and accuracy. As tasks scale from Easy to Hard, the simple heuristic agent begins to fail, particularly against the temporal logic of the **Masked Profile** in Hard variants. This proves the environment is highly robust and requires a true Temporal RL model (like LSTM/PPO integrations) to solve perfectly.

*Designed with 💡 for the OpenEnv Hackathon.*
