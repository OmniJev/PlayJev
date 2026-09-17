// PlayJev hook for taniarascia/sokoban (canvas game, turn based, no game loop, no randomness),
// patched to play the 155 Microban levels with standard push rules (see NOTES.md).
//
// The whole model is one array of strings on the Sokoban instance: sokoban.board[y][x] is one of
// 'empty' | 'wall' | 'block' | 'success_block' | 'void' | 'player' ('void' is a goal square), and
// sokoban.levelMap[y][x] is the level's initial state, consulted to know whether a goal lies under
// the player or a box. A move happens synchronously inside the document keydown handler in
// script.js (it reads event.key), which then calls sokoban.render() to repaint the canvas. Nothing
// is scheduled, so one key press is one complete game step and the canvas is current the moment
// the event returns.
//
// Vendor patch (details in NOTES.md): script.js exposes the module-local instance as
// window.sokoban; utils.js builds the board from levels/microban.js (XSB rows) instead of
// levelOneMap; Sokoban.js keeps a per-level map, sizes the canvas to the level, pushes exactly one
// box (standard rules), and wins when every goal holds a box.
//
// The level is seed % 155 (0-based index, Microban number index + 1). The game has no death and no
// timer, so an episode ends when the level is solved or at the driver's step cap (pj.json max_steps).
(function () {
  const SOLVE_BONUS = 100; // added to the score on the step that completes the level

  const WALL = 'wall', BLOCK = 'block', GOAL_BLOCK = 'success_block', GOAL = 'void', PLAYER = 'player';

  let steps = 0, maxOnGoal = 0, startOnGoal = 0, levelIndex = 0;
  let goals = [], walls = [], goalSet = new Set();

  const sk = () => window.sokoban;
  function cells(grid, pred) {
    const out = [];
    grid.forEach((row, y) => row.forEach((c, x) => { if (pred(c)) out.push([x, y]); }));
    return out;
  }
  function onGoal() {
    let n = 0;
    for (const row of sk().board) for (const c of row) if (c === GOAL_BLOCK) n++;
    return n;
  }
  function player() { const p = sk().findPlayerCoords(); return [p.x, p.y]; }
  function solved() { return goals.length > 0 && onGoal() === goals.length; }
  function repaintBoard() {
    const s = sk();
    s.board.forEach((row, y) => row.forEach((c, x) => s.paintCell(s.context, c, x, y)));
  }

  window.addEventListener('DOMContentLoaded', () => {
    pj.register({
      id: 'sokoban',
      actions: [
        { name: 'up', description: 'move one square up, pushing a box ahead if the square behind it is free', keys: ['ArrowUp'] },
        { name: 'down', description: 'move one square down, pushing a box ahead if the square behind it is free', keys: ['ArrowDown'] },
        { name: 'left', description: 'move one square left, pushing a box ahead if the square behind it is free', keys: ['ArrowLeft'] },
        { name: 'right', description: 'move one square right, pushing a box ahead if the square behind it is free', keys: ['ArrowRight'] },
      ],

      start(seed) {
        if (!window.sokoban) throw new Error('window.sokoban missing: the script.js patch is not in place');
        if (!sk().levelCount || !sk().levelMap) throw new Error('Sokoban instance has no levelCount/levelMap: the Sokoban.js patch is not in place');
        pj.keyTarget = document.body;
        levelIndex = (Number(seed) >>> 0) % sk().levelCount;
        sk().render({ restart: true, level: levelIndex + 1 });  // load the level, size the canvas, paint it

        // Static layout, read once per episode. Goals come from the level map so a goal under the
        // player at the start ('+' in XSB) counts too.
        goals = cells(sk().levelMap, (c) => c === GOAL || c === GOAL_BLOCK);
        goalSet = new Set(goals.map(([x, y]) => x + ',' + y));
        walls = cells(sk().board, (c) => c === WALL);

        steps = 0;
        startOnGoal = onGoal();      // boxes already on goals when the level begins ('*' in XSB)
        maxOnGoal = startOnGoal;
      },

      // One decision = one move. The keydown handler moves and repaints synchronously; one virtual
      // frame is advanced anyway so anything the page might schedule has run before the canvas is read.
      applyAction(i) {
        pj.press(this.actions[i].keys[0], document.body);
        pj.frames(1);
        steps++;
      },

      afterStep() {
        maxOnGoal = Math.max(maxOnGoal, onGoal());
        // On the winning move render() covers the canvas with a black "A Winner is You!" screen.
        // Repaint the solved board with the game's own paintCell so the final frame is still the
        // play area.
        if (solved()) repaintBoard();
      },

      // Boxes the player has put on goals, as a running max so a box pushed off a goal again cannot
      // lower it, plus a bonus on the step that finishes the level. Starts at 0 (boxes on goals when
      // the level begins do not count). Monotone, never negative.
      score() {
        if (!window.sokoban) return 0;
        return (maxOnGoal - startOnGoal) + (solved() ? SOLVE_BONUS : 0);
      },

      // Solving ends the episode; the step cap is the driver's (pj.json max_steps).
      done() { return solved(); },

      canvas() { return sk().canvas; },

      // What a BFS teacher needs. Coordinates are [x, y] with x the column and y the row, the same
      // orientation as the frame. `board` is the standard XSB text form: '#' wall, ' ' floor,
      // '.' goal, '$' box, '*' box on goal, '@' player, '+' player standing on a goal.
      info() {
        const [px, py] = player();
        return {
          level: levelIndex + 1,        // Microban level number, 1..155
          levelIndex: levelIndex,       // seed % 155
          levels: sk().levelCount,
          cell: sk().cell,              // canvas pixels per board cell
          player: [px, py],
          boxes: cells(sk().board, (c) => c === BLOCK || c === GOAL_BLOCK),
          goals: goals,
          walls: walls,
          boxesOnGoal: onGoal(),
          solved: solved(),
          steps: steps,
          board: sk().board.map((row, y) => row.map((c, x) => {
            const g = goalSet.has(x + ',' + y);
            if (c === WALL) return '#';
            if (c === GOAL_BLOCK) return '*';
            if (c === BLOCK) return '$';
            if (c === PLAYER) return g ? '+' : '@';
            return g ? '.' : ' ';
          }).join('')),
        };
      },
    });
  });
})();
