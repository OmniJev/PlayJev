# Task: 2048 teacher

Read docs/TEACHER_BRIEF.md first. Game `2048`: info() has the 4x4 grid (check NOTES.md for orientation: grid[row][col]
matches the frame). Actions up/down/left/right; a move that changes nothing is legal but wastes the step (teacher must
give it zero mass).

Algorithm: expectimax, depth 2 (player move, chance node over empty cells with 2 at 0.9 and 4 at 0.1, player move),
with the standard heuristic on leaves: weighted empty cells, monotonicity along rows and columns, smoothness, and
max tile in a corner (a snake-shaped weight matrix is fine). Pure Python with the grid as a tuple of 16 ints, moves
implemented with a precomputed row-move table (65536 entries or a dict) so a depth-2 search stays around 1 to 3 ms;
if it is slower, restrict chance nodes to at most 6 sampled empty cells. Soft targets: softmax over the four move
values with a temperature that gives about 0.8 to the best move on typical boards (report the temperature), zero for
moves that do not change the board.

Expected level: mean score above 10,000 with regular 1024 and 2048 tiles (random is about 1,000). Report the tile
distribution reached (fraction of episodes reaching 512, 1024, 2048).
