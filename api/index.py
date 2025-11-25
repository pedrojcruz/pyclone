from fastapi import FastAPI

app = FastAPI()

@app.get("/api/index")
def first():
    return {"Hello World, welcome to Vercel!"}