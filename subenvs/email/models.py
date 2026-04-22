from pydantic import BaseModel

class EmailObservation(BaseModel):
    email: str
    step_count: int
    task_id: str

class EmailAction(BaseModel):
    response: str

class EmailState(BaseModel):
    current_email: str
    step_count: int
    done: bool
    task_id: str