from ise_env import ISE

class ISEEasyTask(ISE):
    def __init__(self):
        super().__init__(
            difficulty_level=0.1, 
            max_steps=15, 
            profile_choices=["normal", "adhd", "normal", "adhd", "normal", "adhd"]
        )
