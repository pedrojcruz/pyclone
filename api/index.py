from fastapi import FastAPI

app = FastAPI()

@app.get("/api/index")
def read_root():
    return {"message": "Hello World!"}