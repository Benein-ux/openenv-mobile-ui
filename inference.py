import os
import json
import requests
from openai import OpenAI

# 1. NO FALLBACKS. Do exactly what the validator asks.
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# 2. Your Environment (Safe to keep default)
BASE_URL = os.environ.get("BASE_URL", "https://benein-openenv-mobile-ui.hf.space")
BENCHMARK = "mobile_ui_auditor"

SYSTEM_PROMPT = """
You are an autonomous UI testing agent. 
Respond with a raw JSON object matching the action schema: action_type, target_node_id, input_value, audit_report.
"""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_task(task_id: str, max_steps: int = 10):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    score = 0.0
    
    # Initialize exactly as they requested
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        response = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id}, timeout=30)
        obs = response.json()
    except Exception as e:
        obs = {"error": "Environment offline", "details": str(e)}

    for step in range(1, max_steps + 1):
        steps_taken = step
        user_prompt = f"Task: {task_id}\nObservation:\n{json.dumps(obs)}\n\nNext Action?"
        
        try:
            # This will now hit THEIR proxy, not Hugging Face
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            
            raw_action = completion.choices[0].message.content.strip()
            if "```json" in raw_action:
                raw_action = raw_action.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_action:
                raw_action = raw_action.split("```")[1].split("```")[0].strip()
                
            action_payload = json.loads(raw_action)
            action_str = json.dumps(action_payload, separators=(',', ':'))
            error = None
            
        except Exception as e:
            action_payload = {"action_type": "error"}
            action_str = "error"
            error = str(e)

        try:
            step_resp = requests.post(f"{BASE_URL}/step", json=action_payload, timeout=30).json()
            obs = step_resp.get("observation", {})
            reward = step_resp.get("reward", 0.0)
            done = step_resp.get("done", True)
        except Exception:
            reward = 0.0
            done = True
            
        rewards.append(reward)
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)
        
        if done:
            score = reward
            break
            
    success = score >= 0.5
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    for task in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        run_task(task)
