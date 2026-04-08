import os
import json
import requests
from openai import OpenAI

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

def run_task(task_id: str, max_steps: int = 10):
    # 1. THE FIX: Strip all invisible spaces and newlines!
    # Also prioritized API_KEY so we explicitly hit the validator's proxy.
    raw_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or "dummy_key"
    api_key = raw_key.strip()
    
    raw_base = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
    api_base = raw_base.strip()
    
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct").strip()
    base_url = os.environ.get("BASE_URL", "https://benein-openenv-mobile-ui.hf.space").strip()

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    # 2. SAFE INITIALIZATION
    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
    except Exception as e:
        print(f"[ERROR] OpenAI init failed: {e}", flush=True)
        client = None

    rewards = []
    steps_taken = 0
    score = 0.0

    # Safe Reset
    try:
        response = requests.post(f"{base_url}/reset", params={"task_id": task_id}, timeout=30)
        obs = response.json()
    except Exception as e:
        obs = {"error": "Environment offline", "details": str(e)}

    for step in range(1, max_steps + 1):
        steps_taken = step
        user_prompt = f"Task: {task_id}\nObservation:\n{json.dumps(obs)}\n\nNext Action?"
        
        action_payload = None
        error = None

        # 3. AI Proxy Call
        if client:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
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
                action_payload = json.loads(content)
            except Exception as exc:
                print(f"[DEBUG] Model request failed: {exc}", flush=True)
                error = str(exc)

        # Fallback if AI fails or validator is running a syntax check without keys
        if not action_payload:
            screen = obs.get("current_screen", "")
            if screen == "MainActivity":
                action_payload = {"action_type": "tap", "target_node_id": "btn_settings"}
            elif screen == "SettingsActivity":
                action_payload = {"action_type": "tap", "target_node_id": "switch_dark_mode"}
            else:
                action_payload = {"action_type": "tap", "target_node_id": "btn_home"}
            if not error:
                error = "fallback_used"

        action_str = json.dumps(action_payload, separators=(',', ':'))

        # Safe Step
        try:
            step_resp = requests.post(f"{base_url}/step", json=action_payload, timeout=30).json()
            obs = step_resp.get("observation", {})
            reward = step_resp.get("reward", 0.0)
            done = step_resp.get("done", True)
        except Exception as e:
            reward = 0.0
            done = True
            if error == "fallback_used": error = str(e)

        rewards.append(reward)
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        if done:
            score = reward
            break
            
    success = score >= 0.5
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    for task in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        try:
            run_task(task)
        except Exception as e:
            print(f"[FATAL] Exception in task {task}: {e}", flush=True)
