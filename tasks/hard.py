from ise_env import ISE

class ISEHardTask(ISE):
    def __init__(self):
         super().__init__(
             difficulty_level=1.0, 
             max_steps=30, 
             profile_choices=["normal", "adhd", "masked", "masked"]
         )
