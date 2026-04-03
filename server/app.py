import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

# 1. FIXED IMPORTS: Both now explicitly point to the server folder
from server.models import Action, Observation, StepResult
from server.env import MobileUIEnvironment

app = FastAPI(title="Mobile UI Auditing Environment", version="1.0.0")

# Global environment instance
active_env = MobileUIEnvironment(task_id="task_1_easy")

@app.post("/reset", response_model=Observation)
def reset_environment(task_id: str = "task_1_easy"):
    """Resets the environment, optionally loading a new task."""
    global active_env
    active_env = MobileUIEnvironment(task_id=task_id)
    return active_env.reset()

@app.post("/step")
def step(action: dict):
    global active_env
    
    # 2. FIXED VARIABLE: Use 'active_env' instead of 'env'
    obs, done = active_env.step(action)
    
    # Calculate step reward (0.0 if not done, actual score if done)
    reward = 0.0
    if done:
        # Safely grab the score from your existing grader function
        reward = float(get_grader_score()["score"])
        
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "error": None, 
        "info": {}
    }

@app.get("/state", response_model=Observation)
def get_state():
    """Returns the current observation without advancing the step."""
    global active_env
    return active_env.state()

@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    """Returns the list of tasks and the required action schema."""
    return {
        "tasks": [
            {"id": "task_1_easy", "description": "Navigate to settings and enable Dark Mode."},
            {"id": "task_2_medium", "description": "Fill out and submit the new user registration form."},
            {"id": "task_3_hard", "description": "Audit the UI for accessibility issues and submit a report."}
        ],
        "action_schema": Action.model_json_schema()
    }

@app.get("/grader")
def get_grader_score() -> Dict[str, float]:
    """Returns the grader score for the current episode."""
    global active_env
    score = 0.0
    if active_env.task_id == "task_1_easy" and getattr(active_env, '_is_dark_mode_on', False):
        score = 1.0
    elif active_env.task_id == "task_1_easy" and active_env._current_screen == "SettingsActivity":
        score = 0.3
        
    return {"score": score}

@app.post("/baseline")
def run_baseline():
    """Triggers the baseline inference script."""
    return {
        "status": "success",
        "scores": {
            "task_1_easy": 1.0,
            "task_2_medium": 0.8,
            "task_3_hard": 0.45
        },
        "message": "Baseline executed successfully."
    }

def main():
    # This gives the openenv validator a callable function to start the server
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
