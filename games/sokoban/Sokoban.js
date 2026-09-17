import {
  EMPTY,
  BLOCK,
  SUCCESS_BLOCK,
  VOID,
  PLAYER,
  directions,
  multiplier,
  colors,
} from './constants.js'
import {
  isBlock,
  isTraversible,
  isVoid,
  getX,
  getY,
  generateGameBoard,
  levelCount,
} from './utils.js'

// PlayJev: the canvas is sized to the level, `multiplier` (75 px) per cell, but never wider or
// taller than this. Levels above 12 x 12 get smaller cells.
const maxCanvasSide = 900

class Sokoban {
  constructor({ level }) {
    this.canvas = document.querySelector('canvas')
    this.context = this.canvas.getContext('2d')
    this.levelCount = levelCount

    this.loadLevel(level)
  }

  // PlayJev: replaces the fixed levelOneMap. Sets the board, the level map (what is under the
  // player or a box: goal or floor), the cell size and the canvas size for a 1-based level number.
  loadLevel(level) {
    const { board, levelMap } = generateGameBoard({ level })
    this.level = level
    this.board = board
    this.levelMap = levelMap
    this.goalCount = levelMap.reduce((n, row) => n + row.filter((cell) => isVoid(cell)).length, 0)

    const columns = board[0].length
    const rows = board.length
    this.cell = Math.min(multiplier, Math.floor(maxCanvasSide / Math.max(columns, rows)))
    this.canvas.width = columns * this.cell
    this.canvas.height = rows * this.cell

    this.context.fillStyle = colors.empty
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height)
  }

  // Unchanged drawing at the original 75 px cell; every length below is the original constant
  // times cell / 75 so smaller cells keep the same proportions.
  paintCell(context, cell, x, y) {
    const m = this.cell
    const inset = (m * 5) / 75
    const lineWidth = (m * 10) / 75

    if (cell === 'void' || cell === 'player') {
      const circleSize = cell === 'player' ? (m * 20) / 75 : (m * 10) / 75

      this.context.beginPath()
      this.context.rect(x * m, y * m, m, m)
      this.context.fillStyle = colors.empty.fill
      this.context.fill()

      this.context.beginPath()
      this.context.arc(x * m + m / 2, y * m + m / 2, circleSize, 0, 2 * Math.PI)
      this.context.lineWidth = lineWidth
      this.context.strokeStyle = colors[cell].stroke
      this.context.fillStyle = colors[cell].fill
      this.context.fill()
      this.context.stroke()
    } else {
      this.context.beginPath()
      this.context.rect(x * m + inset, y * m + inset, m - 2 * inset, m - 2 * inset)
      this.context.fillStyle = colors[cell].fill
      this.context.fill()

      this.context.beginPath()
      this.context.rect(x * m + inset, y * m + inset, m - 2 * inset, m - 2 * inset)
      this.context.lineWidth = lineWidth
      this.context.strokeStyle = colors[cell].stroke
      this.context.stroke()
    }
  }

  render(options = {}) {
    if (options.restart) {
      this.loadLevel(options.level || this.level)
      const label = document.querySelector('header p')
      if (label) {
        label.textContent = `Level ${this.level}`
      }
    }
    this.board.forEach((row, y) => {
      row.forEach((cell, x) => {
        this.paintCell(this.context, cell, x, y)
      })
    })

    // Won when every goal holds a box (a goal under the player is still uncovered).
    const boxesOnGoals = this.board.reduce(
      (n, row) => n + row.filter((cell) => cell === SUCCESS_BLOCK).length,
      0,
    )
    const isWin = boxesOnGoals === this.goalCount

    if (isWin) {
      // A winner is you
      this.context.fillStyle = '#111'
      this.context.fillRect(0, 0, this.canvas.width, this.canvas.height)
      this.context.font = 'bold 60px sans-serif'
      this.context.fillStyle = colors.success_block.fill
      this.context.fillText('A Winner is You!', 65, 300)
    }
  }

  findPlayerCoords() {
    const y = this.board.findIndex((row) => row.includes(PLAYER))
    const x = this.board[y].indexOf(PLAYER)

    return {
      x,
      y,
      above: this.board[y - 1][x],
      below: this.board[y + 1][x],
      sideLeft: this.board[y][x - 1],
      sideRight: this.board[y][x + 1],
    }
  }

  movePlayer(playerCoords, direction) {
    // Replace previous spot with initial board state (void or empty)
    this.board[playerCoords.y][playerCoords.x] =
      isVoid(this.levelMap[playerCoords.y][playerCoords.x]) ? VOID : EMPTY

    // Move player
    this.board[getY(playerCoords.y, direction, 1)][getX(playerCoords.x, direction, 1)] = PLAYER
  }

  // Standard Sokoban push: exactly one box moves, and only onto free floor or a free goal.
  // A wall or a second box behind it blocks the move (no chain pushes).
  movePlayerAndBoxes(playerCoords, direction) {
    const newBoxY = getY(playerCoords.y, direction, 2)
    const newBoxX = getX(playerCoords.x, direction, 2)

    if (!isTraversible(this.board[newBoxY][newBoxX])) {
      return
    }

    // Move box
    // If on top of void, make into a success box
    this.board[newBoxY][newBoxX] = isVoid(this.levelMap[newBoxY][newBoxX]) ? SUCCESS_BLOCK : BLOCK
    this.movePlayer(playerCoords, direction)
  }

  move(playerCoords, direction) {
    const { x, y, above, below, sideLeft, sideRight } = playerCoords

    const adjacentCell = {
      [directions.up]: above,
      [directions.down]: below,
      [directions.left]: sideLeft,
      [directions.right]: sideRight,
    }

    if (isTraversible(adjacentCell[direction])) {
      this.movePlayer(playerCoords, direction)
    }

    if (isBlock(adjacentCell[direction])) {
      this.movePlayerAndBoxes(playerCoords, direction)
    }
  }
}

export default Sokoban
