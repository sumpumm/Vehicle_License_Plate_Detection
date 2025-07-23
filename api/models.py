from pydantic import BaseModel

class ModelInput(BaseModel):
    filePath: str
    fileName: str