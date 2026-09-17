// PlayJev hook for jakesgordon/javascript-racer (v4.final.html + common.js, canvas pseudo-3D racer).
//
// All game state is plain globals of the page's inline script: position (camera z on a looping
// track of trackLength units), speed, playerX (-1..1 is the road), segments, cars, currentLapTime,
// lastLapTime, keyLeft/keyRight/keyFaster/keySlower. Nothing is hidden in closures, so the hook
// reads them directly and the vendored files stay untouched.
//
// Loop: Game.run (common.js) loads the two images on real time, then runs frame() on
// requestAnimationFrame with Util.timestamp() = new Date().getTime() and a fixed 1/60 s accumulator,
// so under the shim every pj.frames(k) is k rAF callbacks at 16.67 ms and about k physics updates
// (the integer-millisecond Date alternates 16 and 17 ms, which the accumulator absorbs).
// Keys: keydown/keyup on document, matched by keyCode; the shim's default target bubbles there.
//
// Randomness: resetRoad() (scenery and the 200 traffic cars) runs when the images finish loading,
// before pj.seed, so every page would get the same traffic. start() calls the game's own
// resetRoad() again after seeding, so traffic and scenery depend on the episode seed.
//
// Score: the game has no score, so it is distance travelled along the track in road segments
// (200 units each), as a running maximum so the small push-back on a collision never lowers it.
// The episode ends when the game records a lap (lastLapTime set) or after MAX_STEPS steps.
(function () {
  const MAX_STEPS = 2400;        // 2400 steps x 5 frames = 200 s of game time; a clean lap is about 112 s at top speed
  const LOAD_TIMEOUT_MS = 30000;

  function realSleep(ms) { return new Promise((r) => pj.real.setTimeout(r, ms)); }
  function ready() { return typeof window.segments !== 'undefined' && window.segments.length > 0 && window.trackLength > 0 && window.background && window.sprites; }

  // Distance bookkeeping: position wraps at trackLength; collisions push position back a little.
  let lastPos = 0, dist = 0, maxDist = 0, steps = 0;
  function track() {
    const T = window.trackLength; if (!T) return 0;
    let d = window.position - lastPos;
    if (d < -T / 2) d += T; else if (d > T / 2) d -= T;   // forward wrap / backward push across the line
    dist += d; lastPos = window.position;
    if (dist > maxDist) maxDist = dist;
    return maxDist;
  }
  function playerSegment() { return window.findSegment(window.position + window.playerZ); }

  window.addEventListener('DOMContentLoaded', () => {
    pj.register({
      id: 'racer',
      stepFrames: 5,     // 83 ms: at top speed the car covers 5 segments and steering shifts playerX by 0.17 (the road spans -1..1)
      actions: [
        { name: 'left', description: 'steer to the left', keys: ['ArrowLeft'] },
        { name: 'right', description: 'steer to the right', keys: ['ArrowRight'] },
        { name: 'faster', description: 'press the accelerator', keys: ['ArrowUp'] },
        { name: 'slower', description: 'press the brake', keys: ['ArrowDown'] },
        { name: 'left faster', description: 'accelerate while steering left', keys: ['ArrowLeft', 'ArrowUp'] },
        { name: 'right faster', description: 'accelerate while steering right', keys: ['ArrowRight', 'ArrowUp'] },
      ],
      async start(seed) {
        pj.keyTarget = document.body;
        const t0 = pj.real.Date.now();
        // Game.run waits for two <img> loads (real time); no virtual time is needed for that.
        while (!ready()) {
          if (pj.real.Date.now() - t0 > LOAD_TIMEOUT_MS) throw new Error('racer: game never became ready');
          await realSleep(5);
        }
        // Per-seed traffic and scenery: rebuild the road with the seeded Math.random. The layout
        // (curves, hills, trackLength) is fixed by the game; only sprites and the 200 cars move.
        window.resetRoad();
        window.position = 0; window.speed = 0; window.playerX = 0;
        window.currentLapTime = 0; window.lastLapTime = null;
        lastPos = 0; dist = 0; maxDist = 0; steps = 0;
        pj.frames(2);      // first rendered frame of the new road
      },
      afterStep() { steps++; },
      score() { return Math.round(track() / window.segmentLength * 10) / 10; },
      done() { return window.lastLapTime !== null || steps >= MAX_STEPS; },
      canvas() { return document.getElementById('canvas'); },
      info() {
        const seg = playerSegment(); const idx = seg.index; const S = window.segments; const n = S.length;
        const ahead = [];
        for (let k = 0; k < 60 && ahead.length < 6; k++) {
          const s = S[(idx + k) % n];
          for (const c of s.cars) ahead.push({ dz: Math.round(k * window.segmentLength + (c.z % window.segmentLength) - ((window.position + window.playerZ) % window.segmentLength)), x: Math.round(c.offset * 100) / 100, speed: Math.round(c.speed) });
        }
        const curveAhead = [10, 30, 60].map((k) => Math.round(S[(idx + k) % n].curve * 100) / 100);
        return {
          position: Math.round(window.position), trackLength: window.trackLength, segmentIndex: idx, segmentsTotal: n,
          distance: Math.round(track()), speed: Math.round(window.speed), maxSpeed: window.maxSpeed,
          playerX: Math.round(window.playerX * 1000) / 1000, offRoad: window.playerX < -1 || window.playerX > 1,
          curve: seg.curve, curveAhead, slope: Math.round((seg.p2.world.y - seg.p1.world.y) / window.segmentLength * 1000) / 1000,
          lapTime: Math.round(window.currentLapTime * 10) / 10, lastLapTime: window.lastLapTime,
          carsAhead: ahead, steps,
        };
      },
    });
  });
})();
