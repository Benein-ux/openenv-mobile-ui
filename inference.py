import os
import json
import requests
from openai import OpenAI

# 1. THE FIX: Flip the priority! Grab their API_KEY first. 
# If they don't provide one during a dry-run, fallback to a dummy key so it doesn't crash.
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "dummy_key"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

BASE_URL = os.getenv("BASE_URL", "https://benein-openenv-mobile-ui.hf.space")
BENCHMARK = "mobile_ui_auditor"

SYSTEM_PROMPT = """
You are an autonomous UI testing agent. 
Respond ONLY with a raw JSON object:
{
  "action_type": "...",
  "target_node_id": "...",
  "input_value": "...",
  "audit_report": [...]
}
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

def get_model_message(client: OpenAI, task_id: str, obs: dict) -> dict:
    user_prompt = f"Task: {task_id}\nObservation:\n{json.dumps(obs)}\n\nNext Action?"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        content = (completion.choices[0].message.content or "").strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback to prevent crash, modeled exactly after the official sample script
        screen = obs.get("current_screen", "")
        if screen == "MainActivity":
            return {"action_type": "tap", "target_node_id": "btn_settings"}
        elif screen == "SettingsActivity":
            return {"action_type": "tap", "target_node_id": "switch_dark_mode"}
        else:
            return {"action_type": "tap", "target_node_id": "btn_home"}

def main():
    # Initialize client EXACTLY like the sample
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        
        rewards = []
        steps_taken = 0
        score = 0.0
        
        try:
            response = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id}, timeout=30)
            obs = response.json()
        except Exception as e:
            print(f"[DEBUG] Env reset failed: {e}", flush=True)
            obs = {}

        for step in range(1, 11):
            steps_taken = step
            
            action_payload = get_model_message(client, task_id, obs)
            action_str = json.dumps(action_payload, separators=(',', ':'))
            
            try:
                step_resp = requests.post(f"{BASE_URL}/step", json=action_payload, timeout=30).json()
                obs = step_resp.get("observation", {})
                reward = step_resp.get("reward", 0.0)
                done = step_resp.get("done", True)
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                score = reward
                break
                
        success = score >= 0.5
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
