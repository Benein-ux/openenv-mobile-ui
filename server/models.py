from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- OBSERVATION SPACE ---

class UIElement(BaseModel):
    """Represents a single node in the view hierarchy."""
    node_id: str = Field(description="Unique identifier for the view node")
    class_name: str = Field(description="The type of view (e.g., TextView, Button, ImageView)")
    text: Optional[str] = Field(default=None, description="Visible text on the element")
    content_description: Optional[str] = Field(default=None, description="Accessibility label")
    is_clickable: bool = Field(default=False, description="Whether the element can be tapped")
    is_focusable: bool = Field(default=False, description="Whether the element can receive focus")
    children: List['UIElement'] = Field(default_factory=list, description="Nested child views")

class Observation(BaseModel):
    current_screen: str
    view_tree: UIElement
    system_message: Optional[str] = None
    
    # === NEW: Mobile-realistic observation fields ===
    viewport_height: int = Field(default=1920, description="Simulated screen height in pixels")
    viewport_width: int = Field(default=1080, description="Simulated screen width in pixels")
    scroll_offset: int = Field(default=0, description="Current vertical scroll position in pixels")
    focus_node_id: Optional[str] = Field(default=None, description="Node ID of currently focused element")
    is_loading: bool = Field(default=False, description="Whether the screen is in a loading state")
    # ===============================================

# Resolve forward references for recursive UIElement
UIElement.model_rebuild()

# --- ACTION SPACE ---

class Action(BaseModel):
    """The action the agent chooses to take."""
    action_type: Literal["tap", "input_text", "swipe", "submit_audit"] = Field(
        description="The type of interaction to perform."
    )
    target_node_id: Optional[str] = Field(
        default=None, 
        description="The ID of the UI element to interact with (required for tap/input)."
    )
    input_value: Optional[str] = Field(
        default=None, 
        description="Text to type into a field (required for input_text)."
    )
    audit_report: Optional[List[str]] = Field(
        default=None, 
        description="List of node_ids that fail accessibility standards (required for submit_audit)."
    )

# --- REWARD SPACE ---

class StepResult(BaseModel):
    """The structured response from the step() function."""
    observation: Observation
    reward: float = Field(description="Continuous score from 0.0 to 1.0")
    done: bool = Field(description="True if the task is complete or failed unrecoverably")
    info: dict = Field(description="Additional debugging or grader info")
