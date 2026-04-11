import uvicorn
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

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()
