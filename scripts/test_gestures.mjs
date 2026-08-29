// Standalone Node unit tests + a micro-benchmark for the temporal motion
// engine in js/gestures.js. The module has zero DOM/MediaPipe dependency by
// design (see its header comment), so it runs here with no browser, no
// camera, and no build step — `node scripts/test_gestures.mjs`.
//
// Not wired into a test framework/CI (this project has none — see
// package.json) — this is a focused verification tool for the temporal
// engine specifically, run manually when that code changes.

import assert from 'node:assert/strict';
import {
  LM, LandmarkBuffer, computeTrajectoryFeatures, DynamicHysteresis,
  rescoreWithTrajectory, predictGesture, setAzslModel,
} from '../js/gestures.js';

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log(`  ok  - ${name}`);
}

// 21 landmarks; only indices LandmarkBuffer actually reads (WRIST=0,
// THUMB_TIP=4, INDEX_TIP=8, MIDDLE_MCP=9) carry meaningful values in these
// tests — the rest are inert placeholders. MIDDLE_MCP is kept at a fixed
// offset FROM the wrist (not a fixed absolute position) so a translating
// hand keeps a constant hand-span, matching how a real rigid hand moves —
// scale normalization is supposed to reflect hand size (camera distance),
// not get perturbed by the hand's own translation.
function blankHand() { return handAt(0, 0, 0); }
function handAt(wristX, wristY, wristZ) {
  const pts = [];
  for (let i = 0; i < 21; i++) pts.push({ x: 0, y: 0, z: 0 });
  pts[LM.WRIST] = { x: wristX, y: wristY, z: wristZ };
  pts[LM.MIDDLE_MCP] = { x: wristX, y: wristY - 0.3, z: wristZ };
  return pts;
}

// ---------------------------------------------------------------------- //
console.log('LandmarkBuffer (ring buffer semantics)');
test('count saturates at capacity, never grows past it', () => {
  const buf = new LandmarkBuffer(14);
  for (let i = 0; i < 50; i++) buf.push(blankHand(), i);
  assert.equal(buf.count, 14);
  assert.equal(buf.wristX.length, 14); // backing store never reallocated
});
test('oldest/newest reflect only the most recent `capacity` pushes', () => {
  const buf = new LandmarkBuffer(5);
  for (let i = 0; i < 12; i++) {
    buf.push(handAt(i * 0.01, 0, 0), i);
  }
  // last 5 pushes were i=7..11
  assert.equal(buf.at(0).wristX.toFixed(2), (7 * 0.01).toFixed(2));
  assert.equal(buf.at(4).wristX.toFixed(2), (11 * 0.01).toFixed(2));
});
test('reset() drops history in O(1) without touching the backing arrays', () => {
  const buf = new LandmarkBuffer(14);
  for (let i = 0; i < 20; i++) buf.push(blankHand(), i);
  buf.reset();
  assert.equal(buf.count, 0);
  assert.equal(computeTrajectoryFeatures(buf, false).isDynamic, false);
});

// ---------------------------------------------------------------------- //
console.log('computeTrajectoryFeatures');
test('a still hand reads as not dynamic', () => {
  const buf = new LandmarkBuffer(14);
  for (let i = 0; i < 14; i++) buf.push(blankHand(), i * 16.7);
  const traj = computeTrajectoryFeatures(buf, false);
  assert.equal(traj.isDynamic, false);
});
test('a hand moving steadily downward reads as dynamic with netDy > 0', () => {
  const buf = new LandmarkBuffer(14);
  for (let i = 0; i < 14; i++) {
    buf.push(handAt(0, i * 0.02, 0), i * 16.7); // image-space Y increases downward
  }
  const traj = computeTrajectoryFeatures(buf, false);
  assert.equal(traj.isDynamic, true);
  assert.ok(traj.netDy > 0, `expected netDy > 0, got ${traj.netDy}`);
});
test('a hand oscillating vertically reads as dynamic with multiple sign flips', () => {
  const buf = new LandmarkBuffer(14);
  for (let i = 0; i < 14; i++) {
    buf.push(handAt(0, Math.sin(i) * 0.15, 0), i * 16.7); // oscillates, net displacement ~0
  }
  const traj = computeTrajectoryFeatures(buf, false);
  assert.equal(traj.isDynamic, true);
  assert.ok(traj.verticalSignFlips >= 2, `expected >=2 sign flips, got ${traj.verticalSignFlips}`);
});
test('mirrorX flips the X-axis net delta only', () => {
  const buf = new LandmarkBuffer(14);
  for (let i = 0; i < 14; i++) {
    buf.push(handAt(i * 0.02, 0, 0), i * 16.7);
  }
  const right = computeTrajectoryFeatures(buf, false);
  const mirrored = computeTrajectoryFeatures(buf, true);
  assert.ok(right.netDx > 0);
  assert.ok(mirrored.netDx < 0);
  assert.equal(right.netDy, mirrored.netDy);
});

// ---------------------------------------------------------------------- //
console.log('DynamicHysteresis (150-300ms debounce)');
test('a suggestion only confirms after holdMs of continuous support', () => {
  const h = new DynamicHysteresis(220);
  assert.equal(h.update('Ö', 0), null);      // just proposed
  assert.equal(h.update('Ö', 100), null);    // still within hold window
  assert.equal(h.update('Ö', 219), null);
  assert.equal(h.update('Ö', 220), 'Ö');     // held for exactly holdMs -> confirmed
});
test('a single noisy frame resets the hold window', () => {
  const h = new DynamicHysteresis(220);
  h.update('Ö', 0);
  h.update('Ö', 200);
  h.update(null, 210);      // one noisy frame breaks continuity
  assert.equal(h.update('Ö', 300), null); // window restarted at t=210, only 90ms elapsed
});
test('release back to static also requires holdMs of continuous absence', () => {
  const h = new DynamicHysteresis(220);
  h.update('Ö', 0);
  assert.equal(h.update('Ö', 220), 'Ö');
  h.update(null, 300);
  assert.equal(h.confirmedLabel, 'Ö'); // not yet released
  assert.equal(h.update(null, 519), 'Ö');
  assert.equal(h.update(null, 520), null); // held holdMs -> released
});

// ---------------------------------------------------------------------- //
console.log('rescoreWithTrajectory (spec disambiguation rules)');
test('O + downward motion -> Ö, once hysteresis confirms', () => {
  const h = new DynamicHysteresis(220);
  const traj = { isDynamic: true, netDx: 0.01, netDy: 0.05, verticalSignFlips: 0 };
  let result;
  for (let t = 0; t <= 260; t += 40) {
    result = rescoreWithTrajectory({ label: 'O', confidence: 0.7, candidates: [{ label: 'O', confidence: 0.7 }] }, traj, h, t);
  }
  assert.equal(result.label, 'Ö');
});
test('O with no motion stays O', () => {
  const h = new DynamicHysteresis(220);
  const traj = { isDynamic: false, netDx: 0, netDy: 0, verticalSignFlips: 0 };
  const result = rescoreWithTrajectory({ label: 'O', confidence: 0.7, candidates: [{ label: 'O', confidence: 0.7 }] }, traj, h, 500);
  assert.equal(result.label, 'O');
});
test('U + vertical oscillation -> Ü, once hysteresis confirms', () => {
  const h = new DynamicHysteresis(220);
  const traj = { isDynamic: true, netDx: 0, netDy: 0.001, verticalSignFlips: 3 };
  let result;
  for (let t = 0; t <= 260; t += 40) {
    result = rescoreWithTrajectory({ label: 'U', confidence: 0.65, candidates: [{ label: 'U', confidence: 0.65 }] }, traj, h, t);
  }
  assert.equal(result.label, 'Ü');
});
test('C + downward motion -> Ç, once hysteresis confirms', () => {
  const h = new DynamicHysteresis(220);
  const traj = { isDynamic: true, netDx: 0.005, netDy: 0.04, verticalSignFlips: 0 };
  let result;
  for (let t = 0; t <= 260; t += 40) {
    result = rescoreWithTrajectory({ label: 'C', confidence: 0.68, candidates: [{ label: 'C', confidence: 0.68 }] }, traj, h, t);
  }
  assert.equal(result.label, 'Ç');
});
test('a known-dynamic letter with corroborating motion is boosted above MIN_CONFIDENCE (0.55)', () => {
  const h = new DynamicHysteresis(220);
  const traj = { isDynamic: true, netDx: 0.03, netDy: 0.01, verticalSignFlips: 0 };
  // 0.40 raw confidence would be dropped by predictGesture's MIN_CONFIDENCE
  // gate on its own — this is the exact failure mode reported ("fails
  // completely on dynamic gestures").
  const result = rescoreWithTrajectory({ label: 'Z', confidence: 0.40, candidates: [{ label: 'Z', confidence: 0.40 }] }, traj, h, 0);
  assert.ok(result.confidence > 0.40, `expected a boost, got ${result.confidence}`);
  assert.ok(result.confidence >= 0.55, `expected the boost to clear MIN_CONFIDENCE, got ${result.confidence}`);
});
test('a known-dynamic letter with NO corroborating motion is penalized (likely single-frame noise)', () => {
  const h = new DynamicHysteresis(220);
  const traj = { isDynamic: false, netDx: 0, netDy: 0, verticalSignFlips: 0 };
  const result = rescoreWithTrajectory({ label: 'K', confidence: 0.6, candidates: [{ label: 'K', confidence: 0.6 }] }, traj, h, 0);
  assert.ok(result.confidence < 0.6, `expected a penalty, got ${result.confidence}`);
});
test('a null classifier result passes through untouched and clears hysteresis', () => {
  const h = new DynamicHysteresis(220);
  h.update('Ö', 0); // seed a pending suggestion
  const result = rescoreWithTrajectory({ label: null, confidence: 0, candidates: [] }, { isDynamic: false }, h, 500);
  assert.equal(result.label, null);
  assert.equal(h.pendingLabel, null);
});

// ---------------------------------------------------------------------- //
console.log('predictGesture (end-to-end wiring smoke test, synthetic model)');
test('trajectory rescoring composes correctly through the public predictGesture API', () => {
  // Minimal fake trained model: level1 always dispatches to cluster "0",
  // whose single-layer net (no hidden layers, zero weights) always reports
  // class 'O' with equal logits -> softmax gives ~0.5 confidence for 'O'.
  // That is below CONTROL_OVERRIDE_CONFIDENCE (0.75) so detectControlGesture
  // still runs, and below what a downward-motion 'O'->'Ö' promotion needs to
  // clear MIN_CONFIDENCE unassisted — exercising the exact rescue path.
  const dim = 84;
  const zeroRow = new Array(dim).fill(0);
  const identityish = { mean: new Array(dim).fill(0), std: new Array(dim).fill(1) };
  const fakeCluster = {
    featureIndices: Array.from({ length: dim }, (_, i) => i),
    scaler: identityish,
    model: {
      classes: ['O', 'Ö'],
      outputActivation: 'softmax',
      layers: [{ weights: zeroRow.map(() => [0, 0]), biases: [0, 0] }],
    },
  };
  setAzslModel({
    level1: {
      scaler: identityish,
      model: { classes: ['0'], outputActivation: 'softmax', layers: [{ weights: zeroRow.map(() => [0]), biases: [0] }] },
    },
    clusters: { '0': fakeCluster },
  });

  const landmarks = [];
  for (let i = 0; i < 21; i++) landmarks.push({ x: 0, y: 0, z: 0 });
  landmarks[LM.MIDDLE_MCP] = { x: 0, y: -0.3, z: 0 };

  const hysteresis = new DynamicHysteresis(220);
  let result;
  for (let t = 0; t <= 260; t += 40) {
    result = predictGesture(
      landmarks, false, { x: 0, y: 0 },
      { isDynamic: true, netDx: 0.01, netDy: 0.05, netIndexDx: 0, netIndexDy: 0, verticalSignFlips: 0, speed: 0.05, directionDeg: 90 },
      hysteresis, t
    );
  }
  assert.equal(result.label, 'Ö');
});
test('predictGesture still works with trajectory omitted (back-compat call shape)', () => {
  const landmarks = [];
  for (let i = 0; i < 21; i++) landmarks.push({ x: 0, y: 0, z: 0 });
  landmarks[LM.MIDDLE_MCP] = { x: 0, y: -0.3, z: 0 };
  const result = predictGesture(landmarks, false, { x: 0, y: 0 });
  assert.ok('label' in result && 'confidence' in result);
});

// ---------------------------------------------------------------------- //
console.log('Performance (main-thread cost of the new per-frame work)');
{
  const N = 100000;
  const buf = new LandmarkBuffer(14);
  const hysteresis = new DynamicHysteresis(220);
  for (let i = 0; i < 14; i++) buf.push(blankHand(), i * 16.7); // warm up to full window

  const t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) {
    buf.push(handAt(0, Math.sin(i * 0.1) * 0.02, 0), i * 16.7);
    const traj = computeTrajectoryFeatures(buf, i % 2 === 0);
    rescoreWithTrajectory({ label: 'O', confidence: 0.7, candidates: [{ label: 'O', confidence: 0.7 }] }, traj, hysteresis, i * 16.7);
  }
  const t1 = process.hrtime.bigint();
  const totalMs = Number(t1 - t0) / 1e6;
  const perCallUs = (totalMs * 1000) / N;
  const frameBudgetMs = 1000 / 30; // 30fps
  const pctOfFrameBudget = ((perCallUs / 1000) / frameBudgetMs) * 100;
  console.log(`  ${N.toLocaleString()} push+computeTrajectoryFeatures+rescoreWithTrajectory calls in ${totalMs.toFixed(1)}ms`);
  console.log(`  -> ${perCallUs.toFixed(2)}us/call average (${pctOfFrameBudget.toFixed(3)}% of a ${frameBudgetMs.toFixed(1)}ms/frame budget at 30fps)`);
  assert.ok(perCallUs < 500, `expected comfortably sub-millisecond, got ${perCallUs.toFixed(2)}us`);
  passed++;
}

console.log(`\n${passed} checks passed.`);
