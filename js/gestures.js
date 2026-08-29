// ============================================================================
// PURE GESTURE ENGINE — no DOM access, cannot throw due to missing elements,
// safe to import and call unconditionally at parse time. Extracted out of
// index.html's inline module script so it can be unit-tested standalone in
// plain Node (see scripts/test_gestures.mjs) with zero browser/DOM/camera
// dependency.
// ============================================================================

export const LM = {
  WRIST: 0,
  THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_TIP: 20,
};

// Per-finger (base, j1, j2, tip) landmark quadruples — thumb, index, middle,
// ring, pinky, in this fixed order. Must match FINGERS in
// scripts/extract_azsl_model.py exactly so the 15 joint-angle features line
// up between the trained model and live inference.
const FINGERS = [
  { base: LM.WRIST, j1: LM.THUMB_MCP, j2: LM.THUMB_IP, tip: LM.THUMB_TIP },
  { base: LM.WRIST, j1: LM.INDEX_MCP, j2: LM.INDEX_PIP, tip: LM.INDEX_TIP },
  { base: LM.WRIST, j1: LM.MIDDLE_MCP, j2: LM.MIDDLE_PIP, tip: LM.MIDDLE_TIP },
  { base: LM.WRIST, j1: LM.RING_MCP, j2: LM.RING_PIP, tip: LM.RING_TIP },
  { base: LM.WRIST, j1: LM.PINKY_MCP, j2: LM.PINKY_PIP, tip: LM.PINKY_TIP },
];
const TIP_PAIRS = [
  [LM.THUMB_TIP, LM.INDEX_TIP], [LM.INDEX_TIP, LM.MIDDLE_TIP],
  [LM.MIDDLE_TIP, LM.RING_TIP], [LM.RING_TIP, LM.PINKY_TIP],
];

// Canonical Azerbaijani Latin alphabet (32 letters) — same order used by
// scripts/extract_azsl_model.py and the alphabet reference drawer.
export const AZ_ALPHABET = [
  'A', 'B', 'C', 'Ç', 'D', 'E', 'Ə', 'F', 'G', 'Ğ', 'H', 'X', 'I', 'İ',
  'J', 'K', 'Q', 'L', 'M', 'N', 'O', 'Ö', 'P', 'R', 'S', 'Ş', 'T', 'U',
  'Ü', 'V', 'Y', 'Z',
];

export const LABELS = { SPACE: 'SPACE', DEL: 'DEL' };

const CONTROL_LABEL_META = {
  SPACE: { friendly: 'SPACE',  symbol: '␣' },
  DEL:   { friendly: 'DELETE', symbol: '⌫' },
};

export function labelMeta(label) {
  return CONTROL_LABEL_META[label] || { friendly: label, symbol: label };
}

const MIN_CONFIDENCE = 0.55;
const HEURISTIC_CONF = 0.92;
const ANGLE_STRAIGHT = 155; // finger joint angle (deg) considered "extended"
const ANGLE_FIST = 100;     // finger joint angle (deg) considered "curled"

function vecSub(a, b) { return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }; }
function vecMag(v) { return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
function vecDist(a, b) { return vecMag(vecSub(a, b)); }

function angleBetween(v1, v2) {
  const mags = vecMag(v1) * vecMag(v2) || 1e-6;
  const cos = Math.min(1, Math.max(-1, (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / mags));
  return Math.acos(cos) * (180 / Math.PI);
}

function jointAngleDeg(coords, mcpIdx, pipIdx, tipIdx) {
  const toMcp = vecSub(coords[mcpIdx], coords[pipIdx]);
  const toTip = vecSub(coords[tipIdx], coords[pipIdx]);
  return angleBetween(toMcp, toTip);
}

// Wrist-origin, scale-normalized, then mirrored to a canonical "Right hand"
// orientation for left-hand detections — must match normalize_landmarks()
// in scripts/extract_azsl_model.py exactly (same order of operations).
export function normalizeLandmarks(landmarks, mirrorX) {
  const wrist = landmarks[LM.WRIST];
  const shifted = landmarks.map((p) => vecSub(p, wrist));
  let scale = vecMag(shifted[LM.MIDDLE_MCP]);
  if (scale < 1e-6) scale = 1e-6;
  const normalized = shifted.map((p) => ({ x: p.x / scale, y: p.y / scale, z: p.z / scale }));
  if (mirrorX) normalized.forEach((p) => { p.x = -p.x; });
  return normalized;
}

export function LandmarkSmoother(alpha) {
  this.alpha = alpha || 0.4;
  this.smoothed = null;
}
LandmarkSmoother.prototype.reset = function () { this.smoothed = null; };
LandmarkSmoother.prototype.update = function (landmarks) {
  if (!this.smoothed) {
    this.smoothed = landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z }));
    return this.smoothed;
  }
  const a = this.alpha;
  this.smoothed = landmarks.map((p, i) => ({
    x: a * p.x + (1 - a) * this.smoothed[i].x,
    y: a * p.y + (1 - a) * this.smoothed[i].y,
    z: a * p.z + (1 - a) * this.smoothed[i].z,
  }));
  return this.smoothed;
};

// 10-frame sliding ring buffer over RAW (pre-normalization) wrist position,
// used to feed the trained model's 2-dim velocity feature (see
// buildFeatureVector84 below) — this is a TRAINED INPUT CONTRACT, not just a
// UI convenience: scripts/extract_azsl_model.py computed the exact same
// "position delta / frame-number gap" quantity at training time, so this
// buffer's shape/math must never change without retraining the model.
// Kept separate from the newer LandmarkBuffer/trajectory engine below, which
// is purely an additive post-classification signal and has no such
// constraint.
export function MotionBuffer(size) {
  this.size = size || 10;
  this.buffer = [];
}
MotionBuffer.prototype.reset = function () { this.buffer = []; };
MotionBuffer.prototype.velocityAndPush = function (x, y) {
  let vx = 0, vy = 0;
  if (this.buffer.length) {
    const oldest = this.buffer[0];
    const span = this.buffer.length;
    vx = (x - oldest.x) / span;
    vy = (y - oldest.y) / span;
  }
  this.buffer.push({ x: x, y: y });
  if (this.buffer.length > this.size) this.buffer.shift();
  return { x: vx, y: vy };
};

function fingerState(coords, mcpIdx, pipIdx, tipIdx) {
  const angle = jointAngleDeg(coords, mcpIdx, pipIdx, tipIdx);
  if (angle >= ANGLE_STRAIGHT) return 'extended';
  if (angle <= ANGLE_FIST) return 'curled';
  return 'mid';
}

function thumbExtended(coords, margin) {
  margin = margin || 1.15;
  const tipToPinky = vecDist(coords[LM.THUMB_TIP], coords[LM.PINKY_MCP]);
  const mcpToPinky = vecDist(coords[LM.THUMB_MCP], coords[LM.PINKY_MCP]);
  return tipToPinky > mcpToPinky * margin;
}

function thumbPointingUp(coords, threshold) {
  threshold = threshold || 0.15;
  return coords[LM.THUMB_TIP].y < coords[LM.THUMB_MCP].y - threshold &&
         coords[LM.THUMB_TIP].y < -threshold;
}

function gestureResult(label) { return { label: label, confidence: HEURISTIC_CONF }; }

// The dataset only contains the 32 alphabet letters — there is no trained
// gesture for SPACE/DEL, so those two stay simple geometric heuristics
// layered on top of the trained hierarchical letter classifier below. Both
// checks are built from distances/angles, which are mirror-invariant, so
// they need no handedness correction.
function detectControlGesture(coords) {
  const fs = {
    index: fingerState(coords, LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_TIP),
    middle: fingerState(coords, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP),
    ring: fingerState(coords, LM.RING_MCP, LM.RING_PIP, LM.RING_TIP),
    pinky: fingerState(coords, LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_TIP),
  };
  const thumbOut = thumbExtended(coords);
  const thumbUp = thumbPointingUp(coords);
  const allCurled = fs.index === 'curled' && fs.middle === 'curled' && fs.ring === 'curled' && fs.pinky === 'curled';

  if (thumbOut && thumbUp && allCurled) return gestureResult(LABELS.SPACE);

  // The angle-based "curled" check alone false-fires on some open-palm 'B'
  // shapes (real dataset variation, discovered via scripts validation —
  // moderately bent fingers at some capture angles still read as
  // "curled"). Require the index tip to be CLEARLY the farthest fingertip
  // from the wrist, which a genuine one-finger point has and an open palm
  // does not, as a second independent confirmation.
  const indexDist = vecMag(coords[LM.INDEX_TIP]);
  const indexClearlyFarthest =
    indexDist > vecMag(coords[LM.MIDDLE_TIP]) * 1.2 &&
    indexDist > vecMag(coords[LM.RING_TIP]) * 1.3 &&
    indexDist > vecMag(coords[LM.PINKY_TIP]) * 1.4;

  if (!thumbOut && fs.index === 'extended' && fs.middle === 'curled' &&
      fs.ring === 'curled' && fs.pinky === 'curled' && indexClearlyFarthest) {
    return gestureResult(LABELS.DEL);
  }
  return null;
}

// ------------------------------------------------------------------ //
// Two-level hierarchical classifier against
// public/models/azsl_hierarchical_model.json (Level 1: 6-way
// confusion-cluster dispatcher MLP. Level 2: one specialized MLP per
// cluster — see scripts/extract_azsl_model.py for how these were trained
// and exported; the JS below just replays their forward pass, since no ML
// library is loaded in the browser).
// ------------------------------------------------------------------ //
let AZSL_MODEL = null;

export function setAzslModel(model) { AZSL_MODEL = model; }
export function getAzslModel() { return AZSL_MODEL; }

function jointAngles15(coords) {
  const out = new Array(15);
  const middleDir = vecSub(coords[LM.MIDDLE_PIP], coords[LM.MIDDLE_MCP]);
  for (let i = 0; i < FINGERS.length; i++) {
    const f = FINGERS[i];
    const baseFlex = angleBetween(vecSub(coords[f.base], coords[f.j1]), vecSub(coords[f.j2], coords[f.j1]));
    const tipFlex = angleBetween(vecSub(coords[f.j1], coords[f.j2]), vecSub(coords[f.tip], coords[f.j2]));
    const thisDir = vecSub(coords[f.j2], coords[f.j1]);
    const spread = angleBetween(thisDir, middleDir);
    out[i * 3] = baseFlex; out[i * 3 + 1] = tipFlex; out[i * 3 + 2] = spread;
  }
  return out;
}

function tipDistances4(coords) {
  return TIP_PAIRS.map((pair) => vecDist(coords[pair[0]], coords[pair[1]]));
}

// 84-dim: [63 normalized coords][15 joint angles][4 tip distances][2 velocity].
// Must match build_feature_vector() in scripts/extract_azsl_model.py exactly.
function buildFeatureVector84(coords, velocity) {
  const out = new Array(84);
  let k = 0;
  for (let i = 0; i < coords.length; i++) { out[k++] = coords[i].x; out[k++] = coords[i].y; out[k++] = coords[i].z; }
  const angles = jointAngles15(coords);
  for (let i = 0; i < angles.length; i++) out[k++] = angles[i];
  const dists = tipDistances4(coords);
  for (let i = 0; i < dists.length; i++) out[k++] = dists[i];
  out[k++] = velocity.x; out[k++] = velocity.y;
  return out;
}

function applyScaler(vec, scaler) {
  const out = new Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    const std = scaler.std[i] > 1e-9 ? scaler.std[i] : 1e-9;
    out[i] = (vec[i] - scaler.mean[i]) / std;
  }
  return out;
}

// Replays a trained sklearn MLPClassifier: dense layers with ReLU hidden
// activation, softmax (multi-class) or sigmoid (sklearn's collapsed binary
// case, re-expanded here into two explicit class probabilities) output.
// Returns candidates sorted by confidence, highest first.
function mlpForward(model, inputVec) {
  let activation = inputVec;
  for (let li = 0; li < model.layers.length; li++) {
    const layer = model.layers[li];
    const isOutputLayer = li === model.layers.length - 1;
    const out = new Array(layer.biases.length);
    for (let j = 0; j < layer.biases.length; j++) {
      let sum = layer.biases[j];
      for (let i = 0; i < activation.length; i++) sum += activation[i] * layer.weights[i][j];
      out[j] = sum;
    }
    if (!isOutputLayer) {
      for (let j = 0; j < out.length; j++) out[j] = Math.max(0, out[j]); // ReLU
    }
    activation = out;
  }

  let candidates;
  if (model.outputActivation === 'sigmoid_binary') {
    const p1 = 1 / (1 + Math.exp(-activation[0]));
    candidates = [
      { label: model.classes[0], confidence: 1 - p1 },
      { label: model.classes[1], confidence: p1 },
    ];
  } else {
    const maxLogit = Math.max.apply(null, activation);
    const exps = activation.map((v) => Math.exp(v - maxLogit));
    const sumExp = exps.reduce((a, b) => a + b, 0) || 1e-9;
    candidates = model.classes.map((label, i) => ({ label: label, confidence: exps[i] / sumExp }));
  }
  candidates.sort((a, b) => b.confidence - a.confidence);
  return candidates;
}

function classifyHierarchical(coords, velocity) {
  if (!AZSL_MODEL) return { label: null, confidence: 0, candidates: [] };

  const full84 = buildFeatureVector84(coords, velocity);
  const clusterCandidates = mlpForward(AZSL_MODEL.level1.model, applyScaler(full84, AZSL_MODEL.level1.scaler));
  const clusterEntry = AZSL_MODEL.clusters[String(clusterCandidates[0].label)];
  if (!clusterEntry) return { label: null, confidence: 0, candidates: [] };

  const subInput = clusterEntry.featureIndices.length === full84.length
    ? full84
    : clusterEntry.featureIndices.map((idx) => full84[idx]);
  const letterCandidates = mlpForward(clusterEntry.model, applyScaler(subInput, clusterEntry.scaler));

  return { label: letterCandidates[0].label, confidence: letterCandidates[0].confidence, candidates: letterCandidates.slice(0, 2) };
}

// ============================================================================
// TEMPORAL MOTION BUFFER & TRAJECTORY ANALYSIS ENGINE
//
// The trained classifier above already sees a *little* temporal signal (the
// 2-dim wrist velocity baked into buildFeatureVector84, present because
// Ç,D,G,K,Ö,Ü,Y,Z had genuine multi-frame burst data at training time — see
// scripts/extract_azsl_model.py). In practice that single wrist-only, 2-axis,
// 10-frame signal is too weak on its own to reliably separate a static base
// shape from its motion-marked counterpart (O/Ö, U/Ü, C/Ç), and offers no
// help disambiguating a stroke-shaped letter (Z/D/K/İ) from single-frame
// noise — which is exactly the failure mode reported: dynamic letters read
// as their static look-alike, or drop below MIN_CONFIDENCE entirely.
//
// This engine is a POST-CLASSIFICATION rescoring layer, not a change to the
// trained model's input contract — buildFeatureVector84/MotionBuffer above
// are untouched. It:
//   1. Tracks a short multi-landmark trajectory (LandmarkBuffer) alongside
//      MotionBuffer, independently of it.
//   2. Derives velocity/direction/magnitude features from that trajectory
//      (computeTrajectoryFeatures).
//   3. Applies a small set of spatial-temporal rules to promote a static
//      base letter to its dynamic counterpart when motion corroborates it,
//      and to boost/penalize confidence on the full trained "dynamic" letter
//      set based on whether real motion is actually present
//      (rescoreWithTrajectory).
//   4. Debounces that promotion with a hysteresis window (DynamicHysteresis)
//      so a single noisy frame can't flicker the output between the base and
//      marked letter.
//
// The exact thresholds below (DYNAMIC_MOTION_THRESHOLD in particular) are a
// starting calibration based on typical MediaPipe jitter vs. deliberate
// stroke speed in normalized, scale-invariant units — like the SPACE/DEL
// heuristics above, they are tunable constants, not derived from the
// training data, and should be revisited against real recordings of each
// dynamic letter if precision needs improve.
// ============================================================================

// 12-15 frame sliding window requested; 14 sits in the middle of that band —
// long enough to see a full stroke/oscillation at typical camera frame
// rates (~30fps => ~0.45s of history), short enough to stay responsive and
// keep memory bounded.
const TRAJECTORY_WINDOW = 14;

// Scale-normalized (divided by the wrist->middle-MCP hand span, same scale
// factor normalizeLandmarks() uses) average per-frame displacement of the
// wrist. Below this, motion is treated as hand jitter/tremor; at/above it,
// as a deliberate stroke. Chosen well above typical MediaPipe landmark
// jitter for a still hand (~0.003-0.01/frame in this normalized space) and
// well below a deliberate directional stroke (~0.05+/frame).
const DYNAMIC_MOTION_THRESHOLD = 0.045;

// Hysteresis hold time: a static<->dynamic relabel must be suggested (or
// its absence held) continuously for this long before it is applied, per
// the requested 150-300ms debounce band.
export const DYNAMIC_HYSTERESIS_MS = 220;

// Trajectory-corroboration confidence adjustment applied to any letter in
// KNOWN_DYNAMIC_LETTERS: real accompanying motion boosts (can rescue an
// under-MIN_CONFIDENCE dynamic read), its absence penalizes (flags a
// probable single-frame misfire on a motion-only class).
const TRAJECTORY_CONFIDENCE_BOOST = 0.18;
const TRAJECTORY_CONFIDENCE_PENALTY = 0.25;

// Static base shape -> its motion-marked counterpart, and which rule
// governs the promotion. Keyed by the STATIC letter the classifier reports.
const STATIC_DYNAMIC_PAIRS = {
  O: { dynamic: 'Ö', rule: 'downward' },
  U: { dynamic: 'Ü', rule: 'oscillation' },
  C: { dynamic: 'Ç', rule: 'downward' },
};

// The full set of letters the model was actually trained with multi-frame
// motion bursts for (Ç,D,G,K,Ö,Ü,Y,Z — see the MotionBuffer comment above),
// plus İ per the reported failure set. Any letter in this set gets the
// motion-corroboration confidence adjustment in rescoreWithTrajectory,
// regardless of whether it was reached via a STATIC_DYNAMIC_PAIRS promotion
// or predicted directly by the classifier.
const KNOWN_DYNAMIC_LETTERS = { 'Ç': 1, 'D': 1, 'G': 1, 'K': 1, 'Ö': 1, 'Ü': 1, 'Y': 1, 'Z': 1, 'İ': 1 };

// ---------------------------------------------------------------------- //
// LandmarkBuffer — fixed-capacity ring buffer over 4 tracked points
// (WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP; the last only as a per-frame
// scale reference). Backed by typed arrays sized once at construction and
// written in place: push() never allocates, and older frames are
// overwritten in-place by the ring pointer rather than spliced/shifted out
// — memory is O(window) forever, "purging" is implicit in how a ring
// buffer works rather than a separate GC-generating step.
// ---------------------------------------------------------------------- //
export function LandmarkBuffer(size) {
  const cap = size || TRAJECTORY_WINDOW;
  this.capacity = cap;
  this.wristX = new Float32Array(cap); this.wristY = new Float32Array(cap); this.wristZ = new Float32Array(cap);
  this.thumbX = new Float32Array(cap); this.thumbY = new Float32Array(cap); this.thumbZ = new Float32Array(cap);
  this.indexX = new Float32Array(cap); this.indexY = new Float32Array(cap); this.indexZ = new Float32Array(cap);
  this.midMcpX = new Float32Array(cap); this.midMcpY = new Float32Array(cap); this.midMcpZ = new Float32Array(cap);
  this.t = new Float64Array(cap);
  this.head = 0;   // next write slot
  this.count = 0;  // frames currently held, saturates at capacity
}

LandmarkBuffer.prototype.reset = function () {
  this.head = 0;
  this.count = 0;
  // Stale values beyond `count` are never read (see at()), so there is
  // nothing to zero out — reset is O(1).
};

// landmarks: raw (pre-normalization, pre-smoothing) MediaPipe landmark
// array for this frame; timestampMs: e.g. performance.now().
LandmarkBuffer.prototype.push = function (landmarks, timestampMs) {
  const i = this.head;
  const wrist = landmarks[LM.WRIST], thumb = landmarks[LM.THUMB_TIP],
        index = landmarks[LM.INDEX_TIP], midMcp = landmarks[LM.MIDDLE_MCP];
  this.wristX[i] = wrist.x; this.wristY[i] = wrist.y; this.wristZ[i] = wrist.z;
  this.thumbX[i] = thumb.x; this.thumbY[i] = thumb.y; this.thumbZ[i] = thumb.z;
  this.indexX[i] = index.x; this.indexY[i] = index.y; this.indexZ[i] = index.z;
  this.midMcpX[i] = midMcp.x; this.midMcpY[i] = midMcp.y; this.midMcpZ[i] = midMcp.z;
  this.t[i] = timestampMs;
  this.head = (i + 1) % this.capacity;
  if (this.count < this.capacity) this.count++;
};

// Logical index 0 = oldest held frame, count-1 = newest. Allocates one small
// snapshot object per call — same allocation budget as LandmarkSmoother's
// per-frame output above, called at most once per processed camera frame.
LandmarkBuffer.prototype.at = function (i) {
  const cap = this.capacity;
  const p = (this.head - this.count + i + cap) % cap;
  return {
    wristX: this.wristX[p], wristY: this.wristY[p], wristZ: this.wristZ[p],
    thumbX: this.thumbX[p], thumbY: this.thumbY[p], thumbZ: this.thumbZ[p],
    indexX: this.indexX[p], indexY: this.indexY[p], indexZ: this.indexZ[p],
    midMcpX: this.midMcpX[p], midMcpY: this.midMcpY[p], midMcpZ: this.midMcpZ[p],
    t: this.t[p],
  };
};

const EMPTY_TRAJECTORY = {
  isDynamic: false, speed: 0,
  netDx: 0, netDy: 0, netDz: 0,
  netIndexDx: 0, netIndexDy: 0,
  directionDeg: 0, verticalSignFlips: 0,
};

// Derives velocity/direction/magnitude features from a LandmarkBuffer.
// O(window) per call (window is 12-15 frames — trivial next to the MLP
// forward passes and MediaPipe's own inference that already run every
// frame). mirrorX must match what predictGesture()/normalizeLandmarks()
// used for this frame, so X-axis deltas agree with the canonical
// "right hand" orientation the static/dynamic rules below assume (Y/Z are
// never mirrored, matching normalizeLandmarks()).
export function computeTrajectoryFeatures(buffer, mirrorX) {
  if (!buffer || buffer.count < 4) return EMPTY_TRAJECTORY; // too little history for a meaningful reading

  const n = buffer.count;
  const oldest = buffer.at(0);
  const newest = buffer.at(n - 1);

  // Scale-normalize by the newest frame's hand span so a hand near vs. far
  // from the camera reads the same "speed" for the same physical gesture.
  let scale = Math.sqrt(
    (newest.midMcpX - newest.wristX) * (newest.midMcpX - newest.wristX) +
    (newest.midMcpY - newest.wristY) * (newest.midMcpY - newest.wristY) +
    (newest.midMcpZ - newest.wristZ) * (newest.midMcpZ - newest.wristZ)
  );
  if (scale < 1e-6) scale = 1e-6;

  // Total wrist path length (sum of consecutive-frame displacement) rather
  // than plain endpoint delta — this is what makes oscillation (Ü: hand
  // moves down then up, net displacement ~0) show up as "dynamic" instead
  // of cancelling out.
  let pathLength = 0;
  let verticalSignFlips = 0;
  let prevDy = null;
  for (let i = 1; i < n; i++) {
    const a = buffer.at(i - 1), b = buffer.at(i);
    const dx = b.wristX - a.wristX, dy = b.wristY - a.wristY, dz = b.wristZ - a.wristZ;
    pathLength += Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (Math.abs(dy) > 1e-5) {
      if (prevDy !== null && Math.sign(dy) !== Math.sign(prevDy)) verticalSignFlips++;
      prevDy = dy;
    }
  }
  const speed = (pathLength / scale) / (n - 1);

  let netDx = (newest.wristX - oldest.wristX) / scale;
  const netDy = (newest.wristY - oldest.wristY) / scale;
  const netDz = (newest.wristZ - oldest.wristZ) / scale;
  let netIndexDx = (newest.indexX - oldest.indexX) / scale;
  const netIndexDy = (newest.indexY - oldest.indexY) / scale;
  if (mirrorX) { netDx = -netDx; netIndexDx = -netIndexDx; }

  return {
    isDynamic: speed >= DYNAMIC_MOTION_THRESHOLD,
    speed: speed,
    netDx: netDx, netDy: netDy, netDz: netDz,
    netIndexDx: netIndexDx, netIndexDy: netIndexDy,
    directionDeg: Math.atan2(netDy, netDx) * (180 / Math.PI),
    verticalSignFlips: verticalSignFlips,
  };
}

// ---------------------------------------------------------------------- //
// DynamicHysteresis — confirms a proposed static->dynamic relabel only
// after it has been suggested continuously for `holdMs`, and releases it
// back only after its absence has likewise held for `holdMs`. Symmetric on
// purpose: without the release-side hold, a single noisy frame mid-gesture
// would instantly drop the dynamic label right as the user finishes the
// stroke.
// ---------------------------------------------------------------------- //
export function DynamicHysteresis(holdMs) {
  this.holdMs = holdMs || DYNAMIC_HYSTERESIS_MS;
  this.pendingLabel = undefined; // undefined = "no suggestion tracked yet", distinct from null
  this.pendingSince = 0;
  this.confirmedLabel = null;
}
DynamicHysteresis.prototype.reset = function () {
  this.pendingLabel = undefined;
  this.pendingSince = 0;
  this.confirmedLabel = null;
};
// suggestedLabel: the dynamic label a rule currently proposes, or null if
// none. now: e.g. performance.now(). Returns the currently-confirmed
// override label (or null).
DynamicHysteresis.prototype.update = function (suggestedLabel, now) {
  if (suggestedLabel !== this.pendingLabel) {
    this.pendingLabel = suggestedLabel;
    this.pendingSince = now;
  }
  if (now - this.pendingSince >= this.holdMs) {
    this.confirmedLabel = this.pendingLabel;
  }
  return this.confirmedLabel;
};

// ---------------------------------------------------------------------- //
// rescoreWithTrajectory — the actual disambiguation rules from the spec:
//   O + downward motion            -> Ö
//   U + vertical oscillation       -> Ü
//   C + downward motion            -> Ç
//   any KNOWN_DYNAMIC_LETTERS read -> confidence boosted if corroborated by
//                                      real motion, penalized if not
//     (covers Z/D/K/İ/G/Y directly: their trajectory is not a general
//     linear/curved-stroke classifier here — that needs labeled recordings
//     per letter to calibrate honestly — but genuine accompanying motion
//     vs. none is exactly the corroborating signal that rescues/rejects a
//     borderline read for them too.)
// ---------------------------------------------------------------------- //
export function rescoreWithTrajectory(letterResult, trajectory, hysteresis, now) {
  if (!letterResult || !letterResult.label) {
    if (hysteresis) hysteresis.update(null, now);
    return letterResult;
  }

  let label = letterResult.label;
  let confidence = letterResult.confidence;

  const pair = STATIC_DYNAMIC_PAIRS[label];
  let suggestedOverride = null;
  if (pair && trajectory.isDynamic) {
    if (pair.rule === 'downward' &&
        trajectory.netDy > 0 && Math.abs(trajectory.netDy) >= Math.abs(trajectory.netDx)) {
      suggestedOverride = pair.dynamic;
    } else if (pair.rule === 'oscillation' && trajectory.verticalSignFlips >= 2) {
      suggestedOverride = pair.dynamic;
    }
  }

  const confirmedOverride = hysteresis ? hysteresis.update(suggestedOverride, now) : suggestedOverride;
  if (pair && confirmedOverride === pair.dynamic) {
    label = pair.dynamic;
    // Hysteresis-confirmed trajectory evidence is treated like the other
    // heuristic gestures (SPACE/DEL) above it in confidence terms.
    confidence = Math.max(confidence, HEURISTIC_CONF);
  }

  if (KNOWN_DYNAMIC_LETTERS[label]) {
    confidence = trajectory.isDynamic
      ? Math.min(0.99, confidence + TRAJECTORY_CONFIDENCE_BOOST)
      : Math.max(0, confidence - TRAJECTORY_CONFIDENCE_PENALTY);
  }

  const candidates = (letterResult.candidates || []).map((c) =>
    c.label === letterResult.label ? { label: label, confidence: confidence } : c
  );
  if (!candidates.length) candidates.push({ label: label, confidence: confidence });

  return { label: label, confidence: confidence, candidates: candidates };
}

// SPACE/DEL have no trained representation (dataset = 32 letters only) and
// their handshapes were borrowed from English ASL with no linguistic
// connection to AzSL — testing against real dataset images showed AzSL's
// actual "B" naturally has index/other-finger proportions that can trigger
// the "index-only-extended" DEL heuristic. So the trained classifier goes
// FIRST: only fall back to the geometric heuristic when the trained model
// itself isn't confident about any letter — data wins over guesswork.
const CONTROL_OVERRIDE_CONFIDENCE = 0.75;

// trajectory/hysteresis/now are optional so existing callers (and the unit
// tests) that only care about the static classifier can omit them.
export function predictGesture(landmarksArray, mirrorX, velocity, trajectory, hysteresis, now) {
  const coords = normalizeLandmarks(landmarksArray, mirrorX);
  const v = mirrorX ? { x: -velocity.x, y: velocity.y } : velocity;
  let letterResult = classifyHierarchical(coords, v);

  if (trajectory) {
    letterResult = rescoreWithTrajectory(letterResult, trajectory, hysteresis, now == null ? 0 : now);
  }

  if (letterResult.confidence >= CONTROL_OVERRIDE_CONFIDENCE) return letterResult;

  const control = detectControlGesture(coords);
  if (control) return { label: control.label, confidence: control.confidence, candidates: [control] };

  if (letterResult.confidence >= MIN_CONFIDENCE) return letterResult;
  return { label: null, confidence: letterResult.confidence, candidates: [] };
}
