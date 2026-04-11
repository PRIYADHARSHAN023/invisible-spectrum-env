from fastapi import FastAPI
from ise_env import InvisibleSpectrumEnv

app = FastAPI()

env = InvisibleSpectrumEnv()

@app.post("/reset")
def reset():
    observation = env.reset()
    return {"observation": observation}

@app.post("/step")
def step(action: dict):
    result = env.step(action)
    return result
