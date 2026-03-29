class Grader:
    """
    Evaluates agent performance deterministically.
    """
    @staticmethod
    def calculate_score(is_correct: bool, steps_taken: int, max_steps: int) -> float:
        if not is_correct:
             return 0.2
             
        # Deterministic scoring: more steps = slightly less than 1.0 (efficiency penalty)
        score = 1.0 - (steps_taken / max_steps)
        # Ensure score is tightly bounded [0.0, 1.0]
        return max(0.0, min(1.0, score))
