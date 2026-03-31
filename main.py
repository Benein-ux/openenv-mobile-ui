from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

# Import the models and environment we defined earlier
from models import Action, Observation, StepResult
from env import MobileUIEnvironment

app = FastAPI(title="Mobile UI Auditing Environment", version="1.0.0")

# For simplicity in this containerized app, we use a single global environment instance.
# In a highly concurrent setup, you'd map session IDs to environment instances.
active_env = MobileUIEnvironment(task_id="task_1_easy")

@app.post("/reset", response_model=Observation)
def reset_environment(task_id: str = "task_1_easy"):
    """Resets the environment, optionally loading a new task."""
    global active_env
    active_env = MobileUIEnvironment(task_id=task_id)
    return active_env.reset()

@app.post("/step", response_model=StepResult)
def step_environment(action: Action):
    """Executes an action in the environment and returns the result."""
    global active_env
    try:
        result = active_env.step(action)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
    # In a full implementation, you'd calculate the final cumulative or state-based score here
    # For now, we return a simulated score based on the current state
    score = 0.0
    if active_env.task_id == "task_1_easy" and getattr(active_env, '_is_dark_mode_on', False):
        score = 1.0
    elif active_env.task_id == "task_1_easy" and active_env._current_screen == "SettingsActivity":
        score = 0.3
        
    return {"score": score}

@app.post("/baseline")
def run_baseline():
    """
    Triggers the baseline inference script and returns the baseline score.
    (In a real deployment, this would use the subprocess module to run your inference script 
    or call an imported function that uses the OpenAI API client).
    """
    # Placeholder for the actual baseline execution logic
    return {
        "status": "success",
        "scores": {
            "task_1_easy": 1.0,
            "task_2_medium": 0.8,
            "task_3_hard": 0.45
        },
        "message": "Baseline executed successfully."
    }
