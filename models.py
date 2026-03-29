from pydantic import BaseModel, Field
from typing import Optional, Literal

class Action(BaseModel):
    action_type: Literal["ask_easy", "ask_hard", "classify"]
    value: Optional[Literal["normal", "adhd", "masked"]] = None

class Observation(BaseModel):
    response_time: float
    attention: float
    consistency: float
    difficulty: float
    
class State(BaseModel):
    ground_truth_profile: Literal["normal", "adhd", "masked"]
    steps_taken: int
    terminated: bool
