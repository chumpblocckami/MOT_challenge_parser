from pydantic import BaseModel


class MOTChallenge(BaseModel):
    frame: int
    id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    conf: float = 1.0
    x: int = -1
    y: int = -1
    z: int = -1
