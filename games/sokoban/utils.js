import { EMPTY, WALL, BLOCK, SUCCESS_BLOCK, VOID, PLAYER } from './constants.js'
import { MICROBAN } from './levels/microban.js'

// Helpers
export const isBlock = (cell) => [BLOCK, SUCCESS_BLOCK].includes(cell)
export const isPlayer = (cell) => [PLAYER].includes(cell)
export const isTraversible = (cell) => [EMPTY, VOID].includes(cell)
export const isWall = (cell) => [WALL].includes(cell)
export const isVoid = (cell) => [VOID, SUCCESS_BLOCK].includes(cell)

export const getX = (x, direction, spaces = 1) => {
  if (direction === 'up' || direction === 'down') {
    return x
  }
  if (direction === 'right') {
    return x + spaces
  }
  if (direction === 'left') {
    return x - spaces
  }
}

export const getY = (y, direction, spaces = 1) => {
  if (direction === 'left' || direction === 'right') {
    return y
  }
  if (direction === 'down') {
    return y + spaces
  }
  if (direction === 'up') {
    return y - spaces
  }
}

// PlayJev: levels come from the Microban set (levels/microban.js, XSB rows) instead of levelOneMap.
export const levelCount = MICROBAN.length

// XSB character -> board cell. A '+' (player standing on a goal) is PLAYER on the board while the
// level map below remembers the goal as VOID, the way levelOneMap is consulted for the original level.
const XSB = {
  '#': WALL,
  ' ': EMPTY,
  '.': VOID,
  $: BLOCK,
  '*': SUCCESS_BLOCK,
  '@': PLAYER,
  '+': PLAYER,
}

// Returns { board, levelMap } for a 1-based level number. `board` is the live state, `levelMap` the
// initial state used to know what lies under the player or a box (goal or plain floor). Ragged rows
// are padded with EMPTY: missing cells are floor outside the walls in XSB, and the game draws floor
// outside as plain floor (levelOneMap does the same for its corner cells).
export function generateGameBoard({ level }) {
  const rows = MICROBAN[level - 1]
  if (!rows) {
    throw new Error(`no level ${level}, have 1..${levelCount}`)
  }
  const width = Math.max(...rows.map((row) => row.length))
  const board = rows.map((row) => row.padEnd(width, ' ').split('').map((ch) => XSB[ch]))
  const levelMap = rows.map((row) =>
    row.padEnd(width, ' ').split('').map((ch) => (ch === '+' ? VOID : XSB[ch])),
  )
  return { board, levelMap }
}
