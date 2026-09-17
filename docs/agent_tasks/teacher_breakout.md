# Task: breakout teacher

Read docs/TEACHER_BRIEF.md first. Game `breakout` (jakesgordon): info() has ball position and velocity, paddle x and
width, lives, level, and possibly the brick layout (see games/breakout/NOTES.md). Actions: left, right, stay; one step
= 5 game updates (83 ms). The hook launches the ball itself when it is parked.

Algorithm: when the ball is moving down, extrapolate its path to the paddle's row, reflecting off the left and right
walls, and move the paddle so its centre lands under that x (dead zone of a few pixels so it does not jitter: stay when
within half a paddle width minus the margin). When the ball moves up, drift toward the court centre (or toward the
ball's x) to be ready. Optionally bias the hit point off-centre to steer the ball toward remaining bricks if the
layout is in info(). Soft targets: 0.9 on the chosen action, 0.1 on stay when moving or spread on the two moves
when staying.

Expected level: the hook agent's ball-tracking probe cleared levels (a cleared level is +1000 in the score); the teacher
should clear several levels per episode and rarely lose a life. Report score, levels cleared, lives lost per episode
over 16 episodes, versus random (mean about 740, median 305).
