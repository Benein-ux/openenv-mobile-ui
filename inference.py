import os
import json
import requests
import time

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

def call_llm_proxy(api_base: str, api_key: str, model: str, messages: list) -> dict:
    """
    Direct HTTP call to the LLM proxy - GUARANTEES validator sees the request.
    """
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "openenv-agent/1.0"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 500,
        "stream": False
    }
    
    print(f"[PROXY_HTTP] POST {url} (key_prefix={api_key[:8] if len(api_key) >= 8 else api_key})", flush=True)
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()  # Raise exception for 4xx/5xx
    return response.json()

def run_task(task_id: str, max_steps: int = 10):
    # CRITICAL: Use injected env vars DIRECTLY - NO fallbacks
    api_key = os.environ["API_KEY"].strip()
    api_base = os.environ["API_BASE_URL"].strip()
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct").strip()
    base_url = os.environ.get("BASE_URL", "https://benein-openenv-mobile-ui.hf.space").strip()

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    rewards = []
    steps_taken = 0
    score = 0.0

    # Reset environment
    response = requests.post(f"{base_url}/reset", params={"task_id": task_id}, timeout=30)
    obs = response.json()

    for step in range(1, max_steps + 1):
        steps_taken = step
        user_prompt = f"Task: {task_id}\nObservation:\n{json.dumps(obs)}\n\nNext Action?"
        
        action_payload = None
        error = None

        # === REQUIRED: Call LLM proxy via raw HTTP on EVERY step ===
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            llm_response = call_llm_proxy(api_base, api_key, model_name, messages)
            
            content = llm_response["choices"][0]["message"]["content"].strip()
            
            # Parse JSON (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            action_payload = json.loads(content)
            print(f"[PROXY_HTTP] Step {step}: LLM response parsed successfully", flush=True)
            
        except requests.exceptions.RequestException as http_err:
            # HTTP-level error (connection, timeout, 4xx, 5xx)
            error = f"HTTP_ERROR: {str(http_err)}"
            print(f"[PROXY_HTTP] Step {step} failed: {error}", flush=True)
            
            # Include error in payload to prove proxy was attempted
            action_payload = {
                "action_type": "tap",
                "target_node_id": "btn_home",
                "input_value": "",
                "audit_report": [error]
            }
            
        except Exception as exc:
            # JSON parse error, missing keys, etc.
            error = f"PARSE_ERROR: {str(exc)}"
            print(f"[PROXY_HTTP] Step {step} parse failed: {error}", flush=True)
            
            action_payload = {
                "action_type": "tap",
                "target_node_id": "btn_home",
                "input_value": "",
                "audit_report": [error]
            }

        action_str = json.dumps(action_payload, separators=(',', ':'))

        # Execute environment step
        try:
            step_resp = requests.post(f"{base_url}/step", json=action_payload, timeout=30).json()
            obs = step_resp.get("observation", {})
            reward = step_resp.get("reward", 0.0)
            done = step_resp.get("done", True)
        except Exception as e:
            reward = 0.0
            done = True
            error = f"{error}; ENV_ERROR: {str(e)}" if error else f"ENV_ERROR: {str(e)}"

        rewards.append(reward)
        log_step(step=step, action=action_str, reward=reward, done=done, error=error or "null")

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
            raise  # Re-raise so validator sees the failure
