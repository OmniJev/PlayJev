# Task: racer teacher

Read docs/TEACHER_BRIEF.md first. Game `racer` (jakesgordon/javascript-racer v4): read games/racer/NOTES.md for
info() fields (speed, playerX lateral offset in -1..1, current segment curve, upcoming curves if exposed, nearby
cars with lateral offset and distance, distance travelled) and for what done() means (lap or step horizon).
Actions: left, right, faster, slower, left+faster, right+faster.

Algorithm: keep playerX inside the road (|playerX| < 0.8) and away from cars ahead in the same lane: if a car is
within a lookahead distance and within lateral 0.4 of the player, steer toward the freer side; on a curve steer
against the centrifugal drift the game applies (the game pushes playerX by curve * speed factor each frame, read the
constant); otherwise accelerate. Use slower only when off-road or when a car cannot be avoided in time. Soft targets:
0.8 on the chosen action, 0.2 spread over the other accelerating actions, zero on slower unless it is the choice.

Expected level: the score is distance; the teacher should finish the horizon at high speed with few off-road or
collision slowdowns. Report distance per episode, mean speed, off-road fraction, collisions, versus random.
