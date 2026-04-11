---
title: OpenEnv Mobile UI Auditor
emoji: 📱
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Mobile UI Auditor for OpenEnv

An embodied evaluation environment for testing Multimodal Large Language Models (MLLMs) and RL agents within complex, dynamic Android View Hierarchies. 

Built on the PyTorch OpenEnv framework, this implementation provides a high-fidelity simulation of mobile UI states, including nested accessibility metadata, viewport tracking, and modal pop-up traps. 

### What Makes This Different

Traditional web-crawling benchmarks (like WebArena) ask models to interact with static DOMs. This environment evaluates **what models actually do** when faced with native mobile constraints:
* **Dynamic State Management:** Actions have irreversible consequences. Failing to check a "Terms of Service" box triggers an environment freeze and spawns an `AlertDialog` trap.
* **Spatial Awareness:** The observation space isn't just a text dump; it includes `viewport_width`, `viewport_height`, and active `focus_node_id` metrics, mimicking real UIAutomator dumps.
* **Rigorous Mathematical Grading:** Complex tasks are not graded on binary pass/fail logic. Accessibility Audits are scored on a strict **F1 Score** curve (Precision vs. Recall).

---

## ⚡ Quick Start

You can interact with the environment using standard HTTP requests. No local setup is needed if pointing to a deployed Hugging Face Space.

```python
import requests
import json

BASE_URL = "http://localhost:7860" # Replace with your Space URL

# 1. Reset environment to Task 2 (Registration Trap)
res = requests.post(f"{BASE_URL}/reset", params={"task_id": "task_2_medium"}).json()
print(f"Current Screen: {res['current_screen']}") 
# Output: RegistrationActivity

# 2. Attempt to submit without accepting Terms of Service
action = {"action_type": "tap", "target_node_id": "btn_submit"}
step_res = requests.post(f"{BASE_URL}/step", json=action).json()

# 3. Observe the dynamic state change trap
print(f"Reward: {step_res['reward']}") # Output: 0.001
print(f"New Screen: {step_res['observation']['current_screen']}") # Output: TosDialog
```

---

## 📊 Benchmarking Scenarios

This environment ships with three distinct difficulty tiers, each testing a different capability of the agent.

| Task ID | Name | Objective | Evaluation Metric |
| :--- | :--- | :--- | :--- |
| `task_1_easy` | **Deep UI Navigation** | Navigate from a bloated feed to Settings to toggle Dark Mode. | **Milestone Rewards:** Yields continuous rewards (0.2 -> 0.4 -> 0.99) for locating hidden menus within large, noisy JSON trees. |
| `task_2_medium` | **Modal Traps** | Fill out a multi-field registration form and avoid the TOS trap. | **State Recovery:** Forces the agent to context-switch, dismiss an injected `AlertDialog`, and re-submit the previous form state. |
| `task_3_hard` | **Accessibility Audit** | Scan a 50+ node view tree to find missing `content_description` tags. | **F1 Score:** Mathematically calculated based on True Positives, False Positives, and False Negatives against a ground-truth array. |

---

## 🎮 Action Space

Agents interact with the environment using a strictly typed JSON payload mimicking standard UIAutomator/Appium events.

**Basic Navigation & Interaction**
```json
{"action_type": "tap", "target_node_id": "btn_settings"}
{"action_type": "input_text", "target_node_id": "input_email", "input_value": "agent@openenv.ai"}
{"action_type": "swipe", "direction": "down"}
```

**Complex Auditing**
```json
{"action_type": "submit_audit", "audit_report": ["img_promo_1", "icon_help", "btn_close_ad"]}
```

---

## 👁️ Observation Space

The state is returned as a parsed JSON tree mimicking a mobile view hierarchy, explicitly enriched with device metadata:

* `current_screen`: The name of the active activity/screen (e.g., `FeedActivity`, `TosDialog`).
* `view_tree`: The nested node structure containing `node_id`, `class_name`, `text`, `is_clickable`, and `content_description`.
* `viewport_width` / `viewport_height`: Simulated device dimensions (Default: 1080x1920).
* `scroll_offset`: Current vertical scroll position.
* `focus_node_id`: The currently focused UI element.

---

## ⚙️ Execution Model

The Mobile UI Auditor runs in **synchronous mode** with a single-client architecture:
* **Synchronous Simulation:** The environment only advances when the server receives a `/step` payload. While waiting for the model's action, the UI state is frozen. This ensures deterministic evaluation regardless of LLM inference latency.
* **Stateless Persistence:** All UI states, form data, and dialog pop-ups are maintained natively within the `env.py` execution loop, ensuring safe resets between evaluation episodes.

### Training at Scale
For algorithms requiring 8 rollouts per step (like PPO or GRPO), evaluations will run sequentially on a single instance. For concurrent evaluations, deploy multiple instances via separate Docker containers.

---

## 🤖 Supported Models

This environment is designed to test text-based reasoning and JSON-parsing capabilities of state-of-the-art models. Recommended models for evaluation include:

| Provider | Model | Notes |
| :--- | :--- | :--- |
| **OpenAI** | GPT-4o / GPT-4-Turbo | Excellent JSON adherence and deep-tree reasoning. |
| **Anthropic** | Claude 3.5 Sonnet | Strong context-switching capabilities for Task 2 traps. |
| **Hugging Face** | Qwen 2.5 (72B) | Strong baseline open-source reasoning. |
| **Hugging Face** | Llama 3.3 (70B) | High performance on Task 3 Accessibility Audits. |

---

## ⚠️ Limitations

As a simulated environment, researchers should be aware of the following constraints:
* **No Pixel Rendering:** This environment operates purely on the structural UI tree (JSON/XML). It does not render a visual frame buffer, meaning models relying exclusively on visual pixel inputs (without text parsing) cannot currently be evaluated.
* **Static Viewports:** While `scroll_offset` is tracked, complex dynamic lazy-loading (e.g., infinite scrolling RecyclerViews) is approximated rather than fully simulated.

---

## 🚀 Deployment

**Hugging Face Spaces (Docker)**
This environment is containerized and ready for immediate deployment.
```bash
docker build -t openenv-mobile-ui .
docker run -p 7860:7860 openenv-mobile-ui
```

**Local Testing**
Start the FastAPI environment server locally:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Run the interactive demo to see the modal traps in action:
```bash
python examples/usage.py
```

Run the fully automated evaluation inference pipeline:
```bash
python inference.py
```
