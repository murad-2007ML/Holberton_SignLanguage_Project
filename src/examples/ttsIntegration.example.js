// ============================================================================
// Minimal example: wiring js/gestures.js's per-frame predictions into
// SignLanguageTTS. Not imported anywhere automatically — copy the relevant
// piece into your camera loop (see index.html's onFrame-equivalent code).
// ============================================================================

import { SignLanguageTTS, VOICES } from '../utils/ttsEngine.js';
import { predictGesture } from '../../js/gestures.js';

export function createGestureTTS() {
  const tts = new SignLanguageTTS({
    azureKey: import.meta.env.VITE_AZURE_SPEECH_KEY,
    azureRegion: import.meta.env.VITE_AZURE_SPEECH_REGION,
    voice: VOICES.banu,
    gttsEndpoint: import.meta.env.VITE_GTTS_ENDPOINT || null, // tier 3, optional
  });

  const liveEl = document.getElementById('liveSentence');
  tts.onBufferUpdate((sentence) => { if (liveEl) liveEl.textContent = sentence; });
  tts.onSpeakingStart((text) => console.log('[TTS] speaking:', text));
  tts.onSpeakingEnd(() => console.log('[TTS] done'));
  tts.onError((detail) => console.warn('[TTS] fallback/error:', detail));

  document.getElementById('btnPause')?.addEventListener('click', () => tts.pause());
  document.getElementById('btnResume')?.addEventListener('click', () => tts.resume());
  document.getElementById('btnStop')?.addEventListener('click', () => tts.stop());
  document.getElementById('btnFlush')?.addEventListener('click', () => tts.flush());
  document.getElementById('voiceSelect')?.addEventListener('change', (e) => tts.setVoice(e.target.value));
  document.getElementById('rateSlider')?.addEventListener('input', (e) => tts.setRate(Number(e.target.value)));

  return tts;
}

// Call this once per processed camera frame with that frame's MediaPipe
// landmarks. Every raw prediction — including rapid consecutive repeats
// while a letter is held — is pushed straight through: the engine's own
// dedupe + idle/punctuation buffering (see ttsEngine.js) turns that noisy
// stream into clean spoken sentences, so no separate hold-to-confirm
// gating is needed here.
export function onGestureFrame(tts, landmarksArray, mirrorX, velocity, trajectory, hysteresis, now) {
  const { label } = predictGesture(landmarksArray, mirrorX, velocity, trajectory, hysteresis, now);
  if (label) tts.pushGestureToken(label); // LABELS.SPACE / LABELS.DEL are handled by ttsEngine directly
}

// Manual "speak current sentence now" button, independent of the buffer.
export function speakNow(tts, text) {
  tts.speak(text);
}
