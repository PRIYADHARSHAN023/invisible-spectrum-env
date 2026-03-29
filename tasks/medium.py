from ise_env import ISE

class ISEMediumTask(ISE):
    def __init__(self):
         super().__init__(
             difficulty_level=0.5, 
             max_steps=20, 
             profile_choices=["normal", "adhd", "masked"]
         )
