from fastapi import FastAPI, HTTPException, Request

app = FastAPI()

@app.get("/health")
def health_check():
    return {"message": "central server running"}
