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
    An adaptive heuristic agent that classifies when confident, showing dynamic step counts!
    """
    # Ask minimal questions to gather data. 
    if current_step < 2:
        return Action(action_type="ask_easy")
        
    # Confident Normal Profile (Stable under pressure)
    if obs.attention > 0.75 and obs.consistency > 0.75:
        # If it's been asked at least 6 questions and holds up, it's definitely normal
        if current_step >= 6:
            return Action(action_type="classify", value="normal")
        # In Easy/Medium, we might be confident earlier if stats are extremely high
        if obs.attention > 0.85 and obs.consistency > 0.85 and obs.difficulty < 0.8:
             return Action(action_type="classify", value="normal")
             
    # Confident ADHD Profile (Chaotic from the start)
    if obs.attention < 0.55 and obs.consistency < 0.55:
        return Action(action_type="classify", value="adhd")
        
    # Masked Profile (If it holds up initially, but starts dropping moderately after a few questions)
    if current_step >= 5 and (obs.attention < 0.72 or obs.consistency < 0.72):
        return Action(action_type="classify", value="masked")
        
    # Provide pressure or gather more info if unsure using randomness
    if current_step < 8:
        action_choice = random.choice(["ask_easy", "ask_hard"]) if current_step > 3 else "ask_easy"
        return Action(action_type=action_choice)
        
    # Ultimate fallback guess out of confusion
    return Action(action_type="classify", value="masked")

def run_evaluation():
    print(f"--- Running Inference with {MODEL_NAME} ---")
    print(f"Connecting to: {API_BASE_URL}\n")
    
    tasks_to_run = ["easy", "medium", "hard"]
    total_score = 0.0
    
    for task_name in tasks_to_run:
        print(f"Evaluating Task: {task_name.upper()}")
        env = get_task(task_name)
        
        # Test 5 episodes per task
        task_score = 0.0
        num_episodes = 5
        
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            
            while not done:
                action = heuristic_agent_step(obs, env.steps)
                obs, reward, done, info = env.step(action)
                
            state = env.state()
            is_correct = (info.get("reason") == "correct_classification")
            ep_score = Grader.calculate_score(is_correct, state.steps_taken, env.max_steps)
            task_score += ep_score
            
            print(f"  Ep {ep+1} | Truth: {state.ground_truth_profile:6} | Agents steps: {state.steps_taken:2} | Correct: {is_correct} | Score: {ep_score:.2f}")
            
        avg_task_score = task_score / num_episodes
        print(f"-> Average Score for {task_name.upper()}: {avg_task_score:.2f}\n")
        total_score += avg_task_score
        
    final_score = total_score / len(tasks_to_run)
    print(f"=====================================")
    print(f"FINAL SUBMISSION SCORE: {final_score:.3f}")
    print(f"=====================================")

if __name__ == "__main__":
    run_evaluation()
