import requests
import json

BASE_URL = "http://127.0.0.1:7860" # Or your HuggingFace URL

def demo_environment():
    print("--- Testing Mobile UI Auditor ---")
    
    # 1. Reset the environment (Task 2)
    print("\n[1] Starting Task 2 (Registration with TOS Trap)...")
    res = requests.post(f"{BASE_URL}/reset", params={"task_id": "task_2_medium"}).json()
    print(f"Current Screen: {res['current_screen']}")
    print(f"Viewport Data: {res['viewport_width']}x{res['viewport_height']}")
    
    # 2. Try to submit without checking TOS
    print("\n[2] Attempting to submit without TOS...")
    action = {"action_type": "tap", "target_node_id": "btn_submit"}
    step_res = requests.post(f"{BASE_URL}/step", json=action).json()
    
    # Observe the trap
    print(f"Reward: {step_res['reward']}")
    print(f"New Screen: {step_res['observation']['current_screen']}")
    print(f"System Msg: {step_res['observation']['system_message']}")

if __name__ == "__main__":
    demo_environment()
