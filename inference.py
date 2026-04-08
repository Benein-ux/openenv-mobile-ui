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
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_client():
    api_base = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")

    if not api_base or not api_key:
        raise RuntimeError("Missing API_BASE_URL or API_KEY")

    print(f"[DEBUG] Using API_BASE_URL={api_base}", flush=True)

    return OpenAI(
        base_url=api_base,
        api_key=api_key
    )


def call_llm(client, model, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )

    content = response.choices[0].message.content.strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)


def safe_action_fallback(obs):
    screen = obs.get("current_screen", "")

    if screen == "MainActivity":
        return {"action_type": "tap", "target_node_id": "btn_settings"}
    elif screen == "SettingsActivity":
        return {"action_type": "tap", "target_node_id": "switch_dark_mode"}
    elif screen == "RegistrationActivity":
        return {"action_type": "input_text", "target_node_id": "input_name", "input_value": "John"}
    else:
        return {"action_type": "tap", "target_node_id": "btn_home"}


def run_task(task_id: str, max_steps: int = 10):
    model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    base_url = os.environ.get("BASE_URL", "https://benein-openenv-mobile-ui.hf.space")

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    client = get_client()

    rewards = []
    steps_taken = 0
    score = 0.0

    # Reset environment
    response = requests.post(
        f"{base_url}/reset",
        params={"task_id": task_id},
        timeout=30
    )
    obs = response.json()

    # Agent loop
    for step in range(1, max_steps + 1):
        steps_taken = step

        user_prompt = f"""
Task: {task_id}
Observation:
{json.dumps(obs)}

Next Action:
"""

        try:
            action_payload = call_llm(client, model_name, SYSTEM_PROMPT, user_prompt)
            action_str = json.dumps(action_payload, separators=(',', ':'))
            error = "null"

        except Exception as e:
            print(f"[ERROR] LLM failed: {e}", flush=True)
            action_payload = safe_action_fallback(obs)
            action_str = json.dumps(action_payload)
            error = str(e)

        step_resp = requests.post(
            f"{base_url}/step",
            json=action_payload,
            timeout=30
        ).json()

        obs = step_resp.get("observation", {})
        reward = step_resp.get("reward", 0.0)
        done = step_resp.get("done", True)

        rewards.append(reward)
        log_step(step, action_str, reward, done, error)

        if done:
            score = reward
            break

    success = score >= 0.5
    log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    for task in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        run_task(task)

