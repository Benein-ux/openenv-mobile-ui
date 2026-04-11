---
title: OpenEnv Mobile UI Auditor
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Mobile UI Auditor (PyTorch OpenEnv)

This repository contains a high-fidelity, dynamic Android View Hierarchy simulator designed to benchmark Multimodal LLMs and RL agents on complex mobile automation and accessibility tasks.

## 🧠 Motivation & Real-World Utility
Modern device-controlling agents need to interact with nested UI layouts, understand accessibility metadata, and manage complex state changes. Unlike static web-crawling environments or grid-world toys, this environment simulates true mobile states. Agents must navigate deep JSON/XML view trees, handle modal pop-up interruptions, respect `is_clickable` constraints, and identify missing `content_description` tags, providing immense practical value for automated UI testing.

## 🚀 Environment Features
* **Deep View Hierarchies:** Generates massive JSON-based UI trees simulating modern Android layouts (RecyclerViews, nested ConstraintLayouts, CardViews).
* **Viewport Metadata:** Tracks `viewport_width`, `viewport_height`, `scroll_offset`, and active `focus_node_id` for a realistic, professional-grade agent observation space.
* **Dynamic State Management:** Simulates background state freezes and modal dialogue interruptions that agents must explicitly context-switch to resolve.

## 👁️ Observation Space
The state is returned as a parsed JSON tree mimicking a mobile view hierarchy, enriched with device metadata:
* `current_screen`: The name of the active activity/screen.
* `view_tree`: The nested node structure containing attributes like `node_id`, `class_name`, `text`, `is_clickable`, and `content_description`.
* `viewport_width` / `viewport_height`: Simulated device dimensions.
* `focus_node_id`: The currently focused UI element.
* `system_message`: Textual feedback from the last action.

## 🎯 Benchmark Tasks & Graders

### Task 1 (Easy): Deep UI Navigation
* **Challenge:** The agent starts on a bloated `FeedActivity` filled with noise. It must correctly identify the bottom navigation bar, tap the `Profile` tab, locate the hidden `Settings` gear, and finally toggle the `Dark Mode` switch. 
* **Grader:** Evaluates the agent's ability to filter signal from noise in large JSON trees, awarding partial continuous rewards for successfully traversing the navigation stack.

### Task 2 (Medium): State Management & Modal Traps
* **Challenge:** Fill out a multi-field registration form. 
* **The Trap:** If the agent clicks "Submit" without checking the "Terms of Service" box, the environment dynamically changes state, freezing the background and rendering an `AlertDialog` over the screen. 
* **Grader:** The agent receives minimum rewards until it learns to context-switch, dismiss the dialog, resolve the missing checkbox, and re-submit.

### Task 3 (Hard): Rigorous Accessibility Auditing
* **Challenge:** The agent is fed a massive view tree (50+ nodes) simulating a complex social media feed. Exactly 5 visual elements have been intentionally stripped of their `content_description`. 
* **Grader:** The agent must scan the tree and submit an array of the violating `node_ids`. The agent is graded using a strict **F1 Score** mathematically calculated based on true positives, false positives, and false negatives, ensuring a rigorous statistical evaluation.

## 🛠️ Setup & Usage

### Running via Docker (Hugging Face Spaces)
```bash
# Build the image
docker build -t openenv-mobile-ui .

# Run the container
docker run -p 7860:7860 openenv-mobile-ui
