# Task: invaders teacher

Read docs/TEACHER_BRIEF.md first. Game `invaders` (StrykerKKD/SpaceInvaders): read games/invaders/NOTES.md for
info() fields (player x, alien positions or count and lowest row, bullets, lives/health). Actions: left, right, noop (the ship
auto-fires every 300 ms, so there is no fire action; noop = hold position under a column); one step about 5 frames.

Algorithm: (1) dodge: if an enemy bullet's extrapolated x at the player's row is within the player's half-width
within the next few steps, move away from it (choose the side with more room); (2) otherwise aim: pick the target
column as the nearest alien column (prefer the lowest alien in a column), move toward it, and hold position (noop)
while the player's x is within a tolerance of the target x, since shots fire automatically. Soft targets: 0.8 on the chosen action, the rest spread
over actions that neither walk into a bullet nor waste the step, zero on actions that walk into a bullet.

Expected level: the hook agent's greedy probe cleared all 40 aliens (score 400, the maximum) in about 335 steps;
random scores 187 and dies around step 180. Report score, aliens killed, clear rate and lives lost per episode over 16
episodes, steps to clear, and cost per step.
