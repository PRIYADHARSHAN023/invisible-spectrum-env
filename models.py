from pydantic import BaseModel, Field
from typing import Optional, Literal

class Action(BaseModel):
    action_type: Literal["ask_easy", "ask_hard", "classify"]
    value: Optional[Literal["normal", "adhd", "masked"]] = None

class Observation(BaseModel):
    response_time: float
    attention_score: float
    consistency_score: float
    
class State(BaseModel):
    ground_truth_profile: Literal["normal", "adhd", "masked"]
    steps_taken: int
    terminated: bool
