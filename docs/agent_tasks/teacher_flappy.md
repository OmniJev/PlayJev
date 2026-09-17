# Task: flappy teacher

Read docs/TEACHER_BRIEF.md first. Game `flappy`: info() has bird y and velocity, the next pipe's x and gap (top,
bottom or centre; see games/flappy/NOTES.md for exact fields and units). Actions: flap, wait. One step = 5 physics
ticks (83 ms); the pipe advances 11.1 px per step.

Algorithm: predict the bird's y at the moment it reaches the next pipe's x using the game's gravity and flap impulse
(constants are in js/main.js: gravity, jump velocity, tick), and flap when the predicted y is below the gap centre
minus a margin, else wait. Tune the margin and the target point (slightly below centre is usually best because a flap
is an impulse upward). Handle the case with no pipe yet (hold altitude near mid-screen). Soft targets: 0.9 on the
chosen action, 0.1 on the other, except when one action collides with certainty within the horizon (then 1.0/0.0).

Expected level: the hook agent's one-line rule scored 2 to 3; a tuned predictive rule should pass tens of pipes.
Report mean and max pipes over 16 episodes and the cause of death when it dies (ceiling, floor, top or bottom pipe).
