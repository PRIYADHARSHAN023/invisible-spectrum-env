from .easy import ISEEasyTask
from .medium import ISEMediumTask
from .hard import ISEHardTask
from .grader import Grader

def get_task(task_name: str):
    if task_name == "easy":
         return ISEEasyTask()
    elif task_name == "medium":
         return ISEMediumTask()
    elif task_name == "hard":
         return ISEHardTask()
    else:
         raise ValueError(f"Unknown task: {task_name}")
