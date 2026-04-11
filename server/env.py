import copy
from typing import Dict, Any, List, Optional
from server.models import Observation, Action, StepResult, UIElement

_EPSILON = 0.001

def _clamp_reward(reward: float) -> float:
    """Clamp reward to strictly (0, 1) range as required by validator."""
    return max(_EPSILON, min(1.0 - _EPSILON, reward))

class MobileUIEnvironment:
    def __init__(self, task_id: str = "task_1_easy"):
        self.task_id = task_id
        self.current_step = 0
        self.max_steps = 15

        # Internal state tracking
        self._current_screen = "FeedActivity"
        self._form_state = {"first": "", "last": "", "email": "", "password": ""}
        self._tos_accepted = False
        self._is_dark_mode_on = False
        
        # Ground truth for Task 3 Audit
        self._missing_a11y_nodes = ["img_promo_1", "btn_share_3", "img_avatar_7", "icon_help", "btn_close_ad"]

        # Load the initial view hierarchy
        self._root_view = self._build_initial_ui()

    # ==========================================
    # THE MONSTER UI BUILDERS
    # ==========================================
    def _build_initial_ui(self) -> UIElement:
        """Generates massive, realistic view hierarchies based on the active task."""
        if self.task_id == "task_2_medium":
            self._current_screen = "RegistrationActivity"
            return UIElement(
                node_id="reg_scroll_view", class_name="ScrollView",
                children=[
                    UIElement(node_id="reg_layout", class_name="LinearLayout", children=[
                        UIElement(node_id="header", class_name="TextView", text="Create an Account"),
                        UIElement(node_id="input_first", class_name="EditText", text="", is_focusable=True),
                        UIElement(node_id="input_last", class_name="EditText", text="", is_focusable=True),
                        UIElement(node_id="input_email", class_name="EditText", text="", is_focusable=True),
                        UIElement(node_id="input_password", class_name="EditText", text="", is_focusable=True),
                        UIElement(node_id="chk_terms", class_name="CheckBox", text="I agree to the TOS", is_clickable=True),
                        UIElement(node_id="btn_submit", class_name="Button", text="Register Now", is_clickable=True)
                    ])
                ]
            )
            
        elif self.task_id == "task_3_hard":
            self._current_screen = "ComplexAuditActivity"
            feed_items = []
            for i in range(10):
                avatar_desc = None if f"img_avatar_{i}" in self._missing_a11y_nodes else f"User {i} avatar"
                share_desc = None if f"btn_share_{i}" in self._missing_a11y_nodes else "Share this post"
                
                feed_items.append(
                    UIElement(node_id=f"post_card_{i}", class_name="CardView", children=[
                        UIElement(node_id=f"img_avatar_{i}", class_name="ImageView", content_description=avatar_desc),
                        UIElement(node_id=f"txt_body_{i}", class_name="TextView", text=f"This is post number {i} with lots of text."),
                        UIElement(node_id=f"btn_like_{i}", class_name="ImageButton", content_description="Like post", is_clickable=True),
                        UIElement(node_id=f"btn_share_{i}", class_name="ImageButton", content_description=share_desc, is_clickable=True),
                    ])
                )
                
            feed_items.append(UIElement(node_id="img_promo_1", class_name="ImageView", is_clickable=True)) 
            feed_items.append(UIElement(node_id="icon_help", class_name="ImageView", is_clickable=True)) 
            feed_items.append(UIElement(node_id="btn_close_ad", class_name="ImageButton", is_clickable=True)) 
            
            return UIElement(node_id="audit_root", class_name="FrameLayout", children=feed_items)

        else:
            # Default to Task 1 (Easy) - Deep Navigation Tree
            self._current_screen = "FeedActivity"
            fake_feed = [UIElement(node_id=f"feed_item_{i}", class_name="TextView", text=f"Feed post {i}") for i in range(5)]
            
            return UIElement(
                node_id="root_layout", class_name="ConstraintLayout",
                children=[
                    UIElement(node_id="app_bar", class_name="Toolbar", children=[
                        UIElement(node_id="app_title", class_name="TextView", text="SocialApp")
                    ]),
                    UIElement(node_id="feed_list", class_name="RecyclerView", children=fake_feed),
                    UIElement(node_id="bottom_nav", class_name="BottomNavigationView", children=[
                        UIElement(node_id="nav_home", class_name="Button", text="Home", is_clickable=True),
                        UIElement(node_id="nav_search", class_name="Button", text="Search", is_clickable=True),
                        UIElement(node_id="nav_profile", class_name="Button", text="Profile", content_description="Go to Profile", is_clickable=True)
                    ])
                ]
            )

    def _build_profile_ui(self) -> UIElement:
        return UIElement(
            node_id="profile_layout", class_name="LinearLayout",
            children=[
                UIElement(node_id="txt_username", class_name="TextView", text="JohnDoe99"),
                UIElement(node_id="btn_edit_profile", class_name="Button", text="Edit", is_clickable=True),
                UIElement(node_id="btn_settings", class_name="ImageButton", content_description="App Settings", is_clickable=True)
            ]
        )

    def _build_settings_ui(self) -> UIElement:
        return UIElement(
            node_id="settings_layout", class_name="LinearLayout",
            children=[
                UIElement(node_id="switch_notifications", class_name="Switch", text="Push Notifications", is_clickable=True),
                UIElement(node_id="switch_dark_mode", class_name="Switch", text="Dark Mode", is_clickable=True)
            ]
        )

    def _build_tos_dialog_ui(self) -> UIElement:
        return UIElement(
            node_id="dialog_root", class_name="FrameLayout",
            children=[
                UIElement(node_id="dialog_window", class_name="AlertDialog", children=[
                    UIElement(node_id="txt_tos_warning", class_name="TextView", text="You must accept the Terms of Service to continue."),
                    UIElement(node_id="btn_accept_dialog", class_name="Button", text="Accept & Close", is_clickable=True)
                ])
            ]
        )

    # ==========================================
    # QWEN'S NEW METADATA HELPERS
    # ==========================================
    def _find_focusable_node(self, node: UIElement) -> Optional[UIElement]:
        if getattr(node, 'is_focusable', False) and getattr(node, 'text', None) is not None:
            return node
        for child in getattr(node, 'children', []):
            result = self._find_focusable_node(child)
            if result:
                return result
        return None

    def _update_ui_text(self, node: UIElement, target_id: str, new_text: str) -> bool:
        if node.node_id == target_id:
            node.text = new_text
            return True
        for child in getattr(node, 'children', []):
            if self._update_ui_text(child, target_id, new_text):
                return True
        return False

    def reset(self) -> Observation:
        self.current_step = 0
        self._form_state = {"first": "", "last": "", "email": "", "password": ""}
        self._tos_accepted = False
        self._is_dark_mode_on = False
        self._root_view = self._build_initial_ui()
        return self.state()

    def state(self) -> Observation:
        focus_node = self._find_focusable_node(self._root_view)
        return Observation(
            current_screen=self._current_screen,
            view_tree=copy.deepcopy(self._root_view),
            system_message="Awaiting action." if self.current_step == 0 else None,
            
            # QWEN'S ADDITIONS INTEGRATED HERE:
            viewport_height=1920,
            viewport_width=1080,
            scroll_offset=0,
            focus_node_id=focus_node.node_id if focus_node else None,
            is_loading=False
        )

    # ==========================================
    # THE MONSTER STEP LOGIC
    # ==========================================
    def step(self, action: Action) -> StepResult:
        if isinstance(action, dict):
            class ActionWrapper:
                def __init__(self, d):
                    self.action_type = d.get("action_type")
                    self.target_node_id = d.get("target_node_id")
                    self.input_value = d.get("input_value")
                    self.audit_report = d.get("audit_report")
            action = ActionWrapper(action)

        self.current_step += 1
        reward = 0.001
        done = False
        system_message = "Action executed."

        if self.current_step >= self.max_steps:
            return StepResult(observation=self.state(), reward=_clamp_reward(0.001), done=True, info={"error": "Max steps exceeded."})

        # --- TASK 1: Deep Navigation ---
        if self.task_id == "task_1_easy":
            if action.action_type == "tap":
                if action.target_node_id == "nav_profile" and self._current_screen == "FeedActivity":
                    self._current_screen = "ProfileActivity"
                    self._root_view = self._build_profile_ui()
                    reward = 0.2
                    system_message = "Navigated to Profile Tab."
                elif action.target_node_id == "btn_settings" and self._current_screen == "ProfileActivity":
                    self._current_screen = "SettingsActivity"
                    self._root_view = self._build_settings_ui()
                    reward = 0.4
                    system_message = "Opened Settings."
                elif action.target_node_id == "switch_dark_mode" and self._current_screen == "SettingsActivity":
                    self._is_dark_mode_on = True
                    reward = 0.99
                    done = True
                    system_message = "Dark mode enabled. Sequence complete."
                else:
                    reward = 0.001
                    system_message = f"Invalid tap on {action.target_node_id} in {self._current_screen}."
            else:
                reward = 0.001
                system_message = "Exploration step. Navigate to Profile -> Settings -> Dark Mode."

        # --- TASK 2: Form with Pop-up Trap ---
        elif self.task_id == "task_2_medium":
            if self._current_screen == "TosDialog":
                if action.action_type == "tap" and action.target_node_id == "btn_accept_dialog":
                    self._current_screen = "RegistrationActivity"
                    self._tos_accepted = True
                    self._root_view = self._build_initial_ui() 
                    reward = 0.3
                    system_message = "Dialog dismissed. TOS accepted."
                else:
                    reward = 0.001
                    system_message = "You are trapped in a dialog. Must tap 'btn_accept_dialog'."
            
            elif action.action_type == "input_text" and action.target_node_id and action.input_value:
                field_map = {"input_first": "first", "input_last": "last", "input_email": "email", "input_password": "password"}
                if action.target_node_id in field_map:
                    key = field_map[action.target_node_id]
                    self._form_state[key] = action.input_value
                    self._update_ui_text(self._root_view, action.target_node_id, action.input_value)
                    reward = 0.1
                    system_message = f"Entered text into {action.target_node_id}."
                else:
                    reward = 0.001
            
            elif action.action_type == "tap" and action.target_node_id == "chk_terms":
                self._tos_accepted = not self._tos_accepted
                reward = 0.1
                system_message = f"TOS Checked: {self._tos_accepted}"

            elif action.action_type == "tap" and action.target_node_id == "btn_submit":
                if not self._tos_accepted:
                    self._current_screen = "TosDialog"
                    self._root_view = self._build_tos_dialog_ui()
                    reward = 0.001
                    system_message = "ERROR: TOS not accepted. Dialog popped up."
                elif all(len(v) > 0 for v in self._form_state.values()):
                    reward = 0.99
                    done = True
                    system_message = "Registration successful!"
                else:
                    reward = 0.001
                    system_message = "Validation Error: All text fields are required."

        # --- TASK 3: Rigorous Audit ---
        elif self.task_id == "task_3_hard":
            if action.action_type == "submit_audit" and action.audit_report is not None:
                reported = set(action.audit_report)
                actual = set(self._missing_a11y_nodes)

                true_positives = len(reported.intersection(actual))
                false_positives = len(reported - actual)
                false_negatives = len(actual - reported)

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0.0

                reward = f1_score
                done = True
                system_message = f"Audit submitted. Missing Found: {true_positives}/5. F1: {reward:.2f}"
            else:
                reward = 0.001
                system_message = "Scan the massive JSON tree. Use 'submit_audit' with a list of nodes missing content_descriptions."

        return StepResult(
            observation=self.state(), # Use state() here so the viewport metadata is always included
            reward=_clamp_reward(reward),
            done=done,
            info={"step": self.current_step}
        )
