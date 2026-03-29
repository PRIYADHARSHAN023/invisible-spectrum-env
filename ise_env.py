import random
from models import Action, Observation, State
from typing import Tuple, Dict, Any

class ISE:
    """
    Invisible Spectrum Environment (ISE)
    A simulated behavioral testing environment to detect underlying neurodevelopmental patterns
    (Specifically focusing on Normal, ADHD-like, and Masked profiles).
    """
    def __init__(self, difficulty_level: float = 0.5, max_steps: int = 20, profile_choices: list = None):
        # difficulty_level defines the margin and noise for the generated profiles.
        # 0.0 is Easy, 0.5 is Medium, 1.0 is Hard.
        self.difficulty_level = difficulty_level
        self.max_steps = max_steps
        self.profile_choices = profile_choices if profile_choices is not None else ["normal", "adhd", "masked"]
        self.reset()
        
    def reset(self, profile: str = None) -> Observation:
        """
        Resets the environment and selects a profile (random if not specified).
        Returns the initial observation.
        """
        if profile is None:
            self.profile = random.choice(self.profile_choices)
        else:
            self.profile = profile
            
        self.steps = 0
        self.terminated = False
        
        # Base stats for Masked profile (starts like normal)
        self.masked_attention = random.uniform(0.6, 0.8) if self.difficulty_level > 0.8 else random.uniform(0.85, 0.95)
        self.masked_consistency = random.uniform(0.6, 0.8) if self.difficulty_level > 0.8 else random.uniform(0.85, 0.95)
        
        return self._get_observation(action_difficulty="easy")

    def _get_observation(self, action_difficulty: str) -> Observation:
        # Easy = 0.1, Medium = 0.5, Hard = 1.0
        noise_factor = self.difficulty_level * 0.20 if self.difficulty_level < 0.8 else 0.25
        
        if self.profile == "normal":
            if self.difficulty_level > 0.8: # Hard
                base_rt = random.uniform(0.2, 0.4)
                base_att = random.uniform(0.6, 0.8)
                base_cons = random.uniform(0.6, 0.8)
            elif self.difficulty_level > 0.4: # Medium
                base_rt = random.uniform(0.15, 0.4)
                base_att = random.uniform(0.65, 0.9)
                base_cons = random.uniform(0.65, 0.9)
            else: # Easy
                base_rt = random.uniform(0.1, 0.3)
                base_att = random.uniform(0.8, 1.0)
                base_cons = random.uniform(0.8, 1.0)
            
        elif self.profile == "adhd":
            if self.difficulty_level > 0.8: # Hard
                base_rt = random.uniform(0.3, 0.6)
                base_att = random.uniform(0.4, 0.7)
                base_cons = random.uniform(0.4, 0.7)
            elif self.difficulty_level > 0.4: # Medium
                base_rt = random.uniform(0.35, 0.7)
                base_att = random.uniform(0.3, 0.6)
                base_cons = random.uniform(0.3, 0.6)
            else: # Easy
                base_rt = random.uniform(0.4, 0.8)
                base_att = random.uniform(0.2, 0.5)
                base_cons = random.uniform(0.2, 0.5)
            
        else: # masked
            # Easy mode base values (Distinguishable right away)
            base_rt = random.uniform(0.3, 0.5)
            base_att = random.uniform(0.6, 0.8)
            base_cons = random.uniform(0.6, 0.8)
            
            # Hard mode overrides it strictly to normal initially
            if self.difficulty_level > 0.8:
                base_rt = random.uniform(0.2, 0.4)
                base_att, base_cons = self.masked_attention, self.masked_consistency
            elif self.difficulty_level > 0.4:  # Medium mode
                base_rt = random.uniform(0.25, 0.5)
                base_att = random.uniform(0.5, 0.75)
                base_cons = random.uniform(0.5, 0.75)
                
            # Degradation logic (CRITICAL for Masked profile under pressure)
            if self.steps > 4:
                # SUDDEN Drop for hard mode
                if self.difficulty_level > 0.8:
                     # Delay the drop until step 7 so a basic agent often classifies wrongly as normal too early.
                     if self.steps == 7:
                         self.masked_attention -= random.uniform(0.15, 0.25)
                         self.masked_consistency -= random.uniform(0.15, 0.25)
                     elif self.steps > 7:
                         self.masked_attention -= random.uniform(0.01, 0.05)
                         self.masked_consistency -= random.uniform(0.01, 0.05)
                     base_att = self.masked_attention
                     base_cons = self.masked_consistency
                else:
                     # Gradual for Easy/Medium
                     drop_att = random.uniform(0.1, 0.2)
                     drop_cons = random.uniform(0.1, 0.2)
                     self.masked_attention -= drop_att
                     self.masked_consistency -= drop_cons
                     base_att -= (drop_att * self.steps / 2)
                     base_cons -= (drop_cons * self.steps / 2)
                
            # Further degradation if asked a hard question
            penalty_rt = random.uniform(0.1, 0.3) if action_difficulty == "hard" else 0.0
            penalty_att = random.uniform(0.05, 0.1) if action_difficulty == "hard" else 0.0
            
            base_rt += penalty_rt
            base_att = max(0.0, base_att - penalty_att)
            # Extra variance for masked on hard questions
            if action_difficulty == "hard":
               base_rt += random.uniform(-0.1, 0.2)

        # Apply general noise (clipping back to 0-1 ranges)
        noise_rt = random.uniform(-noise_factor, noise_factor)
        noise_att = random.uniform(-noise_factor, noise_factor)
        noise_cons = random.uniform(-noise_factor, noise_factor)
        
        return Observation(
            response_time=round(max(0.0, min(1.0, base_rt + noise_rt)), 3),
            attention_score=round(max(0.0, min(1.0, base_att + noise_att)), 3),
            consistency_score=round(max(0.0, min(1.0, base_cons + noise_cons)), 3)
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Takes an action and returns (observation, reward, done, info).
        """
        if self.terminated:
            return self._get_observation("easy"), 0.0, True, {"msg": "Episode already terminated."}
            
        self.steps += 1
        reward = 0.0
        info = {}
        
        # Checking for step limits (penalty if taking too long without classifying)
        if self.steps > self.max_steps:
            self.terminated = True
            return self._get_observation("easy"), -0.5, True, {"reason": "max_steps_exceeded"}
            
        if action.action_type == "classify":
            self.terminated = True
            if action.value == self.profile:
                reward = 1.0
                info["reason"] = "correct_classification"
            else:
                reward = -1.0
                info["reason"] = "wrong_classification"
                
            # Final observation on classify doesn't matter much
            obs = self._get_observation("easy") 
            return obs, reward, True, info
            
        elif action.action_type == "ask_easy":
            reward = -0.02
            obs = self._get_observation("easy")
            return obs, reward, False, info
            
        elif action.action_type == "ask_hard":
            reward = -0.02
            obs = self._get_observation("hard")
            return obs, reward, False, info
            
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
            
    def state(self) -> State:
        """
        Returns the hidden ground truth and environment state.
        Used strictly for evaluation/grading, NOT for the agent to use during steps.
        """
        return State(
            ground_truth_profile=self.profile,
            steps_taken=self.steps,
            terminated=self.terminated
        )
