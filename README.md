---
title: OpenEnv Mobile UI Auditor
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# OpenEnv: Mobile UI Navigation & Accessibility Auditing

This repository contains a fully compliant, real-world OpenEnv environment simulating a mobile application's view hierarchy. It is designed to train and evaluate AI agents on practical, day-to-day software testing and accessibility auditing tasks.

## Motivation & Real-World Utility
Modern device-controlling agents need to interact with nested UI layouts, understand accessibility metadata, and fill out forms. Unlike standard grid-world toy environments, this environment accurately models the JSON/XML view trees found in real Android/iOS development. Agents must navigate hierarchies, respect `is_clickable` constraints, and identify missing `content_description` tags, providing immense practical value for automated UI testing.

## Observation Space
The state is returned as a parsed JSON tree of `UIElement` nodes, mimicking a mobile view hierarchy.
* `current_screen`: The name of the active activity/screen.
* `view_tree`: The nested node structure containing attributes like `node_id`, `class_name`, `text`, `is_clickable`, and `content_description`.
* `system_message`: Textual feedback from the last action (e.g., validation errors or navigation updates).

## Action Space
Agents interact with the environment using a strictly typed JSON payload:
* `tap`: Clicks a target `node_id`.
* `input_text`: Enters an `input_value` into a target `node_id`.
* `swipe`: Simulates a scroll event.
* `submit_audit`: Submits an array of `node_id` strings for accessibility evaluation.

## Tasks & Graders

### 1. Enable Dark Mode (Easy)
* **Objective:** Navigate from the `MainActivity` to the `SettingsActivity` and toggle the dark mode switch.
* **Grader:** Yields a partial reward (`0.3`) for finding the settings menu, and full completion (`1.0`) for toggling the switch. Penalizes invalid taps.

### 2. User Registration (Medium)
* **Objective:** Fill out a multi-field registration form (Name, Email, Password) and submit it.
* **Grader:** Yields partial rewards (`0.2`) per accurately filled field and (`0.4`) for successful submission. Penalizes empty submissions triggering validation errors.

### 3. Accessibility Audit (Hard)
* **Objective:** Scan a complex, nested `RecyclerView` feed and identify all interactive elements missing a `content_description` tag.
* **Grader:** The agent submits a list of flagged `node_id`s. The reward (`0.0` - `1.0`) is calculated using the **F1 Score** (harmonic mean of precision and recall) against the ground-truth list of inaccessible nodes, mathematically ensuring a fair continuous reward.

## Setup & Usage

### Running via Docker (Recommended)
This environment is containerized and ready for Hugging Face Spaces deployment.
```bash
# Build the image
docker build -t openenv-mobile-ui .

# Run the container
docker run -p 7860:7860 openenv-mobile-ui
