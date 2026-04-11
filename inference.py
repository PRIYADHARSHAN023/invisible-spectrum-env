import os
import random
from tasks import get_task, Grader
from models import Action

# Simulate OpenEnv expectations reading env variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-heuristic-v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key-for-local")

def heuristic_agent_step(obs, current_step) -> Action:
    """
    Dummy heuristic that uses fixed confidence thresholds and random probes.
    """
    # Confident Normal Profile (Stable under pressure)
    if obs.attention_score > 0.75 and obs.consistency_score > 0.75:
        # If it's been asked at least 6 questions and holds up, it's definitely normal
        if current_step >= 6:
            return Action(action_type="classify", value="normal")
        # In Easy/Medium, we might be confident earlier if stats are extremely high
        if obs.attention_score > 0.85 and obs.consistency_score > 0.85:
             return Action(action_type="classify", value="normal")
             
    # Confident ADHD Profile (Chaotic from the start)
    if obs.attention_score < 0.55 and obs.consistency_score < 0.55:
        return Action(action_type="classify", value="adhd")
        
    # Masked Profile (If it holds up initially, but starts dropping moderately after a few questions)
    if current_step >= 5 and (obs.attention_score < 0.72 or obs.consistency_score < 0.72):
        return Action(action_type="classify", value="masked")
        
    # Provide pressure or gather more info if unsure using randomness
    if current_step < 8:
        action_choice = random.choice(["ask_easy", "ask_hard"]) if current_step > 3 else "ask_easy"
        return Action(action_type=action_choice)
        
    # Ultimate fallback guess out of confusion
    return Action(action_type="classify", value="masked")

def run_evaluation():
    print(f"--- Running Inference with {MODEL_NAME} ---", flush=True)
    print(f"Connecting to: {API_BASE_URL}\n", flush=True)
    
    tasks_to_run = ["easy", "medium", "hard"]
    total_score = 0.0
    
    for task_name in tasks_to_run:
        print(f"Evaluating Task: {task_name.upper()}", flush=True)
        print(f"[START] task={task_name}", flush=True)
        
        env = get_task(task_name)
        obs = env.reset()
        done = False
        
        while not done:
            action = heuristic_agent_step(obs, env.steps)
            obs, reward, done, info = env.step(action)
            print(f"[STEP] step={env.steps} reward={reward}", flush=True)
            
        state = env.state()
        is_correct = (info.get("reason") == "correct_classification")
        avg_score = Grader.calculate_score(is_correct, state.steps_taken, env.max_steps)
        
        print(f"[END] task={task_name} score={avg_score:.2f} steps={state.steps_taken}", flush=True)
        print(f"-> Score for {task_name.upper()}: {avg_score:.2f}\n", flush=True)
        total_score += avg_score
        
    final_score = total_score / len(tasks_to_run)
    print(f"=====================================", flush=True)
    print(f"FINAL SUBMISSION SCORE: {final_score:.3f}", flush=True)
    print(f"=====================================", flush=True)

if __name__ == "__main__":
    run_evaluation()
