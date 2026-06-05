from fastapi import FastAPI

app = FastAPI(title="PREVIA API")

@app.get("/")
def read_root():
    return {"message": "PREVIA API is running"}
