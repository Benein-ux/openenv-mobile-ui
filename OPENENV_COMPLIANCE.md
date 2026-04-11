# OpenEnv Compliance & Evaluation Matrix

This document outlines how the **Mobile UI Auditor** environment strictly adheres to the PyTorch OpenEnv design philosophy, architectural requirements, and reward grading standards.

## 1. Architectural Compliance
The environment is built to be a drop-in benchmark for OpenEnv agents.

* **Stateless API Compatibility:** The environment exposes standard `/reset` and `/step` endpoints.
* **Deterministic Execution:** Given the same sequence of actions, the environment transitions through the exact same view hierarchy states and modal traps.
* **Dockerized & Isolated:** The environment is fully self-contained within a Docker container, running a lightweight FastAPI server to ensure secure, isolated agent evaluation without external dependencies.

## 2. State & Observation Space Completeness
To accurately benchmark multimodal and reasoning agents, the environment provides a mathematically complete observation space mimicking real-world device automation.

* **Hierarchical State:** Returns deeply nested JSON trees (simulating Android XML view trees).
* **Viewport Context:** Tracks dynamic mobile metadata including `viewport_width` (1080), `viewport_height` (1920), and `scroll_offset`.
* **Focus Tracking:** Dynamically tracks the `focus_node_id` to evaluate an agent's spatial awareness.

## 3. Continuous & Bounded Reward Shaping (Evaluation)
As per OpenEnv requirements, all task scores are strictly clamped to the `(0, 1)` bounds. The environment utilizes rigorous statistical grading rather than binary pass/fail metrics.

### Task 1: Navigation Sequence
* **Metric:** Milestone-based partial rewards.
* **Implementation:** The agent receives `0.2` for locating the hidden profile tab, `0.4` for finding settings, and `0.99` for completing the sequence. This prevents sparse-reward failure loops.

### Task 2: Modal Trap Resolution
* **Metric:** State-recovery evaluation.
* **Implementation:** If the agent triggers the "Terms of Service" trap, it receives a minimum score (`0.001`) until it successfully context-switches, reads the `AlertDialog` state, dismisses it, and resolves the missing boolean flag. 

### Task 3: Accessibility Audit (F1 Score)
* **Metric:** Statistical Precision & Recall.
* **Implementation:** The environment contains a ground-truth array of nodes intentionally missing `content_description` tags. When the agent submits its `audit_report` array, the environment calculates True Positives, False Positives, and False Negatives. The final reward is the exact **F1 Score** of the audit, ensuring a mathematically continuous evaluation of the agent's accuracy.

## 4. Fallback & Crash Safety
* **Robust Error Handling:** Invalid JSON schemas, hallucinatory `node_id` targets, and unsupported action types do not crash the container. They are caught, returning a clamped `0.001` reward alongside a contextual `system_message` to guide the agent's next step.
