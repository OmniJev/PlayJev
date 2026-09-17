"""Snake teacher: BFS to the food over free cells; if no path, move to the neighbour with the largest
reachable free region. Soft targets: 1.0 mass split over moves that keep the shortest distance, a small
share for other safe moves, zero for moves into a wall or the body.

info() from games/snake/pj_hook.js: grid rows joined by '|', '#' wall or body, 'F' food, '.' empty;
head {row, col}. The head cell itself is '#'."""
from __future__ import annotations
from collections import deque
from . import Teacher

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class SnakeTeacher(Teacher):
    def reset(self):
        self.prev_head = None

    def _bfs(self, grid, start):
        R, C = len(grid), len(grid[0]); dist = {start: 0}; q = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in DIRS.values():
                n = (r + dr, c + dc)
                if 0 <= n[0] < R and 0 <= n[1] < C and n not in dist and grid[n[0]][n[1]] != "#":
                    dist[n] = dist[(r, c)] + 1; q.append(n)
        return dist

    def act(self, obs: dict) -> list[float]:
        info = obs["info"]; grid = info["grid"].split("|"); h = info["head"]
        head = (h["row"], h["col"]); n = len(self.actions)
        food = tuple(info["food"]) if info.get("food") else None
        # direction we are travelling (the game ignores a 180 degree turn, so that move wastes a step)
        heading = None
        if self.prev_head is not None:
            d = (head[0] - self.prev_head[0], head[1] - self.prev_head[1])
            heading = next((k for k, v in DIRS.items() if v == d), None)
        self.prev_head = head
        safe, dist_to_food = {}, {}
        food_dist = self._bfs(grid, food) if food else {}
        for name, (dr, dc) in DIRS.items():
            if heading and name == OPPOSITE[heading]:
                continue
            cell = (head[0] + dr, head[1] + dc)
            if not (0 <= cell[0] < len(grid) and 0 <= cell[1] < len(grid[0])) or grid[cell[0]][cell[1]] == "#":
                continue
            safe[name] = len(self._bfs(grid, cell))  # reachable region size
            if cell in food_dist:
                dist_to_food[name] = food_dist[cell]
        probs = [0.0] * n
        if not safe:  # every move dies; spread over legal-looking moves so the target is still a distribution
            for name in DIRS:
                if not (heading and name == OPPOSITE[heading]):
                    probs[self.idx[name]] = 1.0
        elif dist_to_food:
            best = min(dist_to_food.values())
            winners = [k for k, v in dist_to_food.items() if v == best]
            for k in winners: probs[self.idx[k]] = 0.9 / len(winners)
            others = [k for k in safe if k not in winners]
            for k in others: probs[self.idx[k]] = 0.1 / len(others)
            if not others:
                for k in winners: probs[self.idx[k]] = 1.0 / len(winners)
        else:  # no path to food: maximise room
            best = max(safe.values()); winners = [k for k, v in safe.items() if v == best]
            for k in winners: probs[self.idx[k]] = 1.0 / len(winners)
        s = sum(probs)
        return [p / s for p in probs]
