import io
from pydantic import BaseModel, Field, validator, ConfigDict


class PipeInput(BaseModel):
    image_path: str = Field(..., min_length=1, max_length=500)
    num_faces: int = Field(-1)

    @validator('image_path')
    def image_path_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Image path cannot be empty')
        return v.strip()

class MeshOutput(BaseModel):
    mesh: bytes
