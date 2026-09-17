"""Teacher policies. Each teacher reads the hook's info() (internal game state the model never sees)
and returns a probability vector over the game's actions. Registry keyed by game id."""
from __future__ import annotations
import importlib

_REGISTRY: dict[str, str] = {
    "snake": "playjev.teachers.snake:SnakeTeacher",
    "2048": "playjev.teachers.2048:G2048Teacher",
    "tetris": "playjev.teachers.tetris:TetrisTeacher",
    "mario": "playjev.teachers.mario:MarioTeacher",
    "breakout": "playjev.teachers.breakout:BreakoutTeacher",
    "invaders": "playjev.teachers.invaders:InvadersTeacher",
    "racer": "playjev.teachers.racer:RacerTeacher",
    "flappy": "playjev.teachers.flappy:FlappyTeacher",
    "pacman": "playjev.teachers.pacman:PacmanTeacher",
    "sokoban": "playjev.teachers.sokoban:SokobanTeacher",
}


def make_teacher(game_id: str, actions: list[dict]):
    if game_id not in _REGISTRY:
        raise KeyError(f"no teacher for {game_id}; known: {sorted(_REGISTRY)}")
    mod, cls = _REGISTRY[game_id].split(":")
    return getattr(importlib.import_module(mod), cls)(actions)


class Teacher:
    """Stateful per episode: call reset() on a new episode, act(obs) each step."""

    def __init__(self, actions: list[dict]):
        self.actions = actions
        self.idx = {a["name"]: a["index"] for a in actions}

    def reset(self):
        pass

    def act(self, obs: dict) -> list[float]:
        raise NotImplementedError

    def giveup(self) -> bool:
        """True when the teacher knows the episode can no longer be won (a deadlocked Sokoban board, for example).
        The collector then ends the episode early instead of recording hundreds of fallback labels."""
        return False

    @staticmethod
    def one_hot(n: int, i: int) -> list[float]:
        p = [0.0] * n; p[i] = 1.0; return p
