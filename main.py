from fastapi import FastAPI, Request
import uvicorn
from tasks import get_task
from models import Action

app = FastAPI()
current_env = None

print("--- Invisible Spectrum Environment API Initializing ---")

@app.on_event("startup")
async def startup_event():
    print("FastAPI Application Startup Complete on Port 7860")

@app.post("/reset")
async def reset(request: Request):
    global current_env
    try:
        body = await request.json()
        task_name = body.get("task", "easy")
    except Exception:
        task_name = "easy"
        
    try:
        current_env = get_task(task_name)
    except ValueError:
        current_env = get_task("easy")
        
    obs = current_env.reset()
    return obs.model_dump()

@app.post("/step")
async def step(action: Action):
    global current_env
    if current_env is None:
        current_env = get_task("easy")
        current_env.reset()
        
    obs, reward, done, info = current_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

# Standard health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Invisible Spectrum Environment API Running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
