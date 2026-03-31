import copy
from typing import Dict, Any, List
from models import Observation, Action, StepResult, UIElement

class MobileUIEnvironment:
    def __init__(self, task_id: str = "task_1_easy"):
        self.task_id = task_id
        self.current_step = 0
        self.max_steps = 15
        
        # Internal state tracking
        self._current_screen = "MainActivity"
        self._form_state = {"name": "", "email": "", "password": ""}
        self._missing_a11y_nodes = ["img_profile_avatar", "btn_favorite_post"]
        
        # Load the initial view hierarchy based on the task
        self._root_view = self._build_initial_ui()

    def _build_initial_ui(self) -> UIElement:
        if self.task_id == "task_2_medium":
            self._current_screen = "RegistrationActivity"
            return UIElement(
                node_id="reg_layout", class_name="LinearLayout",
                children=[
                    UIElement(node_id="input_name", class_name="EditText", text="", is_focusable=True),
                    UIElement(node_id="input_email", class_name="EditText", text="", is_focusable=True),
                    UIElement(node_id="input_password", class_name="EditText", text="", is_focusable=True),
                    UIElement(node_id="btn_submit", class_name="Button", text="Register", is_clickable=True)
                ]
            )
        elif self.task_id == "task_3_hard":
            self._current_screen = "FeedActivity"
            return UIElement(
                node_id="feed_layout", class_name="RecyclerView",
                children=[
                    # Missing content_description!
                    UIElement(node_id="img_profile_avatar", class_name="ImageView", is_clickable=True), 
                    UIElement(node_id="text_post_body", class_name="TextView", text="Check out my new app project!"),
                    # Missing content_description!
                    UIElement(node_id="btn_favorite_post", class_name="ImageButton", is_clickable=True),
                    # Has content_description (Correct)
                    UIElement(node_id="btn_share", class_name="ImageButton", content_description="Share post", is_clickable=True)
                ]
            )
        else:
            # Default to Task 1 (Easy) UI from previous step
            self._current_screen = "MainActivity"
            return UIElement(node_id="root", class_name="FrameLayout", children=[...]) # Truncated for brevity

    def reset(self) -> Observation:
        self.current_step = 0
        self._form_state = {"name": "", "email": "", "password": ""}
        self._root_view = self._build_initial_ui()
        return self.state()

    def state(self) -> Observation:
        return Observation(
            current_screen=self._current_screen,
            view_tree=copy.deepcopy(self._root_view),
            system_message="Awaiting action." if self.current_step == 0 else None
        )

    def _update_ui_text(self, node: UIElement, target_id: str, new_text: str) -> bool:
        """Recursively updates the text of a specific UI node."""
        if node.node_id == target_id:
            node.text = new_text
            return True
        for child in node.children:
            if self._update_ui_text(child, target_id, new_text):
                return True
        return False

    def step(self, action: Action) -> StepResult:
        self.current_step += 1
        reward = 0.0
        done = False
        system_message = "Action executed."

        if self.current_step >= self.max_steps:
            return StepResult(observation=self.state(), reward=0.0, done=True, info={"error": "Max steps exceeded."})

        # --- Task 2 (Medium): Form Filling ---
        if self.task_id == "task_2_medium":
            if action.action_type == "input_text" and action.target_node_id and action.input_value:
                field_map = {"input_name": "name", "input_email": "email", "input_password": "password"}
                
                if action.target_node_id in field_map:
                    key = field_map[action.target_node_id]
                    self._form_state[key] = action.input_value
                    self._update_ui_text(self._root_view, action.target_node_id, action.input_value)
                    reward = 0.2  # Partial reward for filling a field
                    system_message = f"Entered text into {action.target_node_id}."
                else:
                    reward = -0.1
                    system_message = "Invalid input target."

            elif action.action_type == "tap" and action.target_node_id == "btn_submit":
                if all(len(v) > 0 for v in self._form_state.values()):
                    reward = 0.4  # Final reward for successful submission (0.2 * 3 + 0.4 = 1.0 total)
                    done = True
                    system_message = "Registration successful!"
                else:
                    reward = -0.1
                    system_message = "Validation Error: All fields are required."

        # --- Task 3 (Hard): Accessibility Audit ---
        elif self.task_id == "task_3_hard":
            if action.action_type == "submit_audit" and action.audit_report is not None:
                reported = set(action.audit_report)
                actual = set(self._missing_a11y_nodes)
                
                true_positives = len(reported.intersection(actual))
                false_positives = len(reported - actual)
                false_negatives = len(actual - reported)
                
                # Calculate precision and recall
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                
                # Calculate F1 Score for the reward (0.0 to 1.0)
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0.0
                    
                reward = round(f1_score, 2)
                done = True
                system_message = f"Audit submitted. Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {reward:.2f}"
            else:
                reward = -0.05  # Slight penalty for wasting steps looking around
                system_message = "Exploration step recorded. Use 'submit_audit' when ready."

        return StepResult(
            observation=Observation(
                current_screen=self._current_screen,
                view_tree=copy.deepcopy(self._root_view),
                system_message=system_message
            ),
            reward=reward,
            done=done,
            info={"step": self.current_step}
        )
