import os
import json
import requests
from openai import OpenAI
from pydantic import ValidationError

# Ensure OPENAI_API_KEY is set in your environment variables
client = OpenAI()
BASE_URL = "http://localhost:7860"

# The system prompt teaches the LLM how to interact with the environment
SYSTEM_PROMPT = """
You are an autonomous UI testing agent. Your goal is to complete the assigned task by interacting with a mobile app's view hierarchy.
You will receive a JSON representation of the current screen. 

You must respond with a raw JSON object representing your next action. Do not include markdown formatting or explanations.
The JSON must strictly match this schema:
{
  "action_type": "tap" | "input_text" | "swipe" | "submit_audit",
  "target_node_id": "string (optional, required for tap/input_text)",
  "input_value": "string (optional, required for input_text)",
  "audit_report": ["list of strings (optional, required for submit_audit)"]
}
"""

def run_agent_on_task(task_id: str, max_steps: int = 10) -> float:
    print(f"\n--- Starting {task_id} ---")
    
    # Reset the environment and get the initial observation
    response = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    response.raise_for_status()
    observation = response.json()
    
    for step in range(max_steps):
        # Format the observation for the LLM
        user_prompt = f"Current Task: {task_id}\nObservation:\n{json.dumps(observation, indent=2)}\n\nWhat is your next action?"
        
        # Query the OpenAI API
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Use a fast, capable model for the baseline
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # Keep it deterministic
        )
        
        # Parse the action
        raw_action = completion.choices[0].message.content.strip()
        try:
            # Strip potential markdown blocks if the LLM disobeys the prompt
            if raw_action.startswith("```json"):
                raw_action = raw_action[7:-3].strip()
            
            action_payload = json.loads(raw_action)
            print(f"Step {step + 1} Action: {action_payload['action_type']} on {action_payload.get('target_node_id')}")
            
        except json.JSONDecodeError:
            print(f"Failed to parse LLM output: {raw_action}")
            break

        # Send the action to the environment
        step_resp = requests.post(f"{BASE_URL}/step", json=action_payload)
        step_result = step_resp.json()
        
        observation = step_result["observation"]
        
        if step_result["done"]:
            print(f"Task finished at step {step + 1}.")
            break
            
    # Get the final grader score
    grader_resp = requests.get(f"{BASE_URL}/grader")
    score = grader_resp.json().get("score", 0.0)
    print(f"Final Score for {task_id}: {score}")
    return score

if __name__ == "__main__":
    tasks = ["task_1_easy", "task_2_medium", "task_3_hard"]
    scores = {}
    
    for task in tasks:
        scores[task] = run_agent_on_task(task)
        
    print("\n=== BASELINE RESULTS ===")
    for t, s in scores.items():
        print(f"{t}: {s}")
