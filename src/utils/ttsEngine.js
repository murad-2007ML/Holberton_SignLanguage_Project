// ============================================================================
// SignLanguageTTS — gesture-stream buffering + 3-tier Azerbaijani
// text-to-speech for the AzSL translator.
//
// Engine hierarchy (each tier is only attempted if the one above it throws).
// EVERY tier speaks Azerbaijani (az-AZ) or it doesn't speak at all — there
// is intentionally no "generic"/default-locale fallback anywhere in this
// file. If tiers 1-3 all fail, _synthesize() logs an explicit error and
// stays silent rather than letting the browser or an engine substitute
// English, Turkish, or any other language.
//   1. Azure Cognitive Services Speech SDK, pinned to az-AZ-BanuNeural or
//      az-AZ-BabekNeural (see ALLOWED_AZURE_VOICES — setVoice() rejects
//      anything else). Requires azureKey/azureRegion (config, or the
//      VITE_AZURE_SPEECH_KEY / VITE_AZURE_SPEECH_REGION env vars).
//   2. A free, keyless audio stream, always requesting Azerbaijani
//      explicitly: Google Translate's public translate_tts endpoint with
//      tl=az (hardcoded, never derived from user input), played back via
//      an <audio src="..."> element. config.gttsEndpoint can override this
//      with a custom REST endpoint (string URL or (text, voice) => URL) —
//      e.g. a self-hosted Edge-TTS proxy — as long as it returns az-AZ
//      audio bytes; this module cannot inspect opaque audio to verify that.
//   3. Native window.speechSynthesis, but ONLY if the browser actually
//      exposes a voice whose .lang starts with "az" (case-insensitive).
//      Most Windows/Chrome installs have NO Azerbaijani voice pack, and
//      speechSynthesis will silently substitute the system default
//      (en-US) instead of erroring if you ask for one anyway — so this
//      tier must verify a real az voice exists and skip itself entirely
//      rather than ever calling .speak() with a mismatched voice (the
//      actual root cause of AzSL text being read with an English accent).
//
// This module owns its own gesture buffer: the ML pipeline fires a raw
// prediction on every processed frame ('S','S','A','L','A','M', ...), so
// pushGestureToken() is deliberately independent of any "hold-to-confirm"
// gating the caller may already do — it does its own dedupe + idle/
// punctuation-based sentence flushing so it stays correct even fed a raw,
// noisy per-frame stream directly.
// ============================================================================

const DEFAULT_DEDUPE_WINDOW_MS = 300;
const DEFAULT_IDLE_FLUSH_MS = 1500;
const PUNCTUATION_TRIGGERS = new Set(['.', '!', '?']);
const MIN_RATE = 0.5;
const MAX_RATE = 2.0;

export const VOICES = {
  banu: 'az-AZ-BanuNeural',
  babek: 'az-AZ-BabekNeural',
};

// Only these two are ever accepted as the Azure voice — never validated
// against a pattern, because a pattern like /^az-AZ/ would happily accept
// a future non-Azerbaijani voice Microsoft ships under a similar-looking
// name. An explicit allowlist is the only way to actually guarantee this.
const ALLOWED_AZURE_VOICES = new Set(Object.values(VOICES));

function resolveVoiceName(candidate, fallback) {
  if (typeof candidate === 'string' && ALLOWED_AZURE_VOICES.has(candidate)) return candidate;
  if (candidate != null) {
    console.error(
      `[SignLanguageTTS] Rejected voice "${candidate}" — only ${[...ALLOWED_AZURE_VOICES].join(', ')} ` +
      `are allowed, to guarantee Azerbaijani-only output. Staying on "${fallback}".`
    );
  }
  return fallback;
}

function clampRate(value) {
  const n = Number(value);
  if (Number.isNaN(n)) return 1.0;
  return Math.min(MAX_RATE, Math.max(MIN_RATE, n));
}

// \p{L}/\p{N} (Unicode letter/number categories) match every Azerbaijani
// character — ə, ı, İ, ş, ğ, ç, ö, ü in both cases — without needing to
// enumerate them, and are robust to whatever normalization form the ML
// pipeline's tokens arrive in. Only a small speakable punctuation set
// survives alongside them; everything else (control chars, stray symbols
// gesture noise can occasionally emit) is stripped before it ever reaches
// an engine.
export function sanitizeForSpeech(text) {
  if (text == null) return '';
  return String(text)
    .normalize('NFC')
    .replace(/[^\p{L}\p{N}\s.,!?;:'"()-]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function escapeXml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function buildSsml(text, voiceName, rate) {
  const pct = Math.round((rate - 1) * 100);
  const rateAttr = `${pct >= 0 ? '+' : ''}${pct}%`;
  return (
    `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="az-AZ">` +
    `<voice name="${voiceName}"><prosody rate="${rateAttr}">${escapeXml(text)}</prosody></voice></speak>`
  );
}

export class SignLanguageTTS {
  constructor(config = {}) {
    this.config = {
      // import.meta.env is Vite-injected; falls back to config for callers
      // that already have the key/region from elsewhere (e.g. a server).
      azureKey: config.azureKey || import.meta.env?.VITE_AZURE_SPEECH_KEY || null,
      azureRegion: config.azureRegion || import.meta.env?.VITE_AZURE_SPEECH_REGION || null,
      gttsEndpoint: config.gttsEndpoint || null, // optional override; default is Google Translate TTS (tier 2)
      dedupeWindowMs: config.dedupeWindowMs ?? DEFAULT_DEDUPE_WINDOW_MS,
      idleFlushMs: config.idleFlushMs ?? DEFAULT_IDLE_FLUSH_MS,
      spaceToken: config.spaceToken || 'SPACE',
      deleteToken: config.deleteToken || 'DEL',
    };

    this.voiceName = resolveVoiceName(config.voice, VOICES.banu);
    this.rate = clampRate(config.rate ?? 1.0);

    this.buffer = '';
    this.lastToken = null;
    this.lastTokenAt = 0;
    this.idleTimer = null;

    this.isSpeaking = false;
    this.isPaused = false;
    this.destroyed = false;

    this._listeners = { speakingStart: [], speakingEnd: [], bufferUpdate: [], error: [] };

    this._azure = null; // { sdk, speechConfig, synthesizer?, player? }
    this._azureReady = null; // memoized init Promise<boolean>

    this._browserVoice = undefined; // undefined = not checked yet, null = checked & absent
    this._activeUtterance = null;
    this._activeAudio = null;
    this._activeObjectUrl = null;
    this._activeSettle = null; // force-resolves whichever tier promise is currently in flight, used by stop()
    this._speakingStartEmitted = false;
    this._speakingEndEmitted = false;
    this._synthesisSeq = 0; // bumped per _synthesize() call so a stale/interrupted call's
    this._activeSynthesisSeq = 0; // forced settle can never fire events for the call that superseded it
  }

  // ---------------------------------------------------------------- //
  // Event subscriptions — each returns an unsubscribe function.
  // ---------------------------------------------------------------- //
  onSpeakingStart(cb) { return this._subscribe('speakingStart', cb); }
  onSpeakingEnd(cb) { return this._subscribe('speakingEnd', cb); }
  onBufferUpdate(cb) { return this._subscribe('bufferUpdate', cb); }
  onError(cb) { return this._subscribe('error', cb); }

  _subscribe(event, cb) {
    if (typeof cb !== 'function') return () => {};
    this._listeners[event].push(cb);
    return () => {
      const i = this._listeners[event].indexOf(cb);
      if (i !== -1) this._listeners[event].splice(i, 1);
    };
  }

  _emit(event, payload) {
    for (const cb of this._listeners[event].slice()) {
      try { cb(payload); } catch (err) { console.error(`[SignLanguageTTS] "${event}" listener threw`, err); }
    }
  }

  // ---------------------------------------------------------------- //
  // Gesture stream ingestion
  // ---------------------------------------------------------------- //
  pushGestureToken(token) {
    if (this.destroyed || token == null) return;
    const raw = String(token);
    const now = Date.now();

    // Consecutive-duplicate noise filter: a held gesture re-fires the same
    // label every frame, so collapse repeats within the dedupe window into
    // a single token. Refreshing lastTokenAt on every repeat (rather than
    // only on the first) means a gesture held well past the window still
    // reads as "one long hold", not a slow drip of accepted duplicates.
    if (raw === this.lastToken && (now - this.lastTokenAt) < this.config.dedupeWindowMs) {
      this.lastTokenAt = now;
      return;
    }
    this.lastToken = raw;
    this.lastTokenAt = now;

    if (raw === this.config.spaceToken) {
      this._appendToBuffer(' ');
      return;
    }
    if (raw === this.config.deleteToken) {
      this.buffer = this.buffer.slice(0, -1);
      this._onBufferChanged();
      return;
    }

    const clean = sanitizeForSpeech(raw);
    if (!clean) return;
    this._appendToBuffer(clean);

    if (PUNCTUATION_TRIGGERS.has(clean) || /[.!?]$/.test(clean)) {
      this.flush();
    }
  }

  _appendToBuffer(fragment) {
    this.buffer += fragment;
    this._onBufferChanged();
  }

  _onBufferChanged() {
    this._emit('bufferUpdate', this.buffer);
    this._scheduleIdleFlush();
  }

  _scheduleIdleFlush() {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = null;
    if (!this.buffer.trim()) return;
    this.idleTimer = setTimeout(() => this.flush(), this.config.idleFlushMs);
  }

  // ---------------------------------------------------------------- //
  // Buffer control
  // ---------------------------------------------------------------- //
  flush() {
    if (this.idleTimer) { clearTimeout(this.idleTimer); this.idleTimer = null; }
    const text = sanitizeForSpeech(this.buffer);
    this.clearBuffer();
    if (text) this._synthesize(text);
  }

  clearBuffer() {
    if (this.idleTimer) { clearTimeout(this.idleTimer); this.idleTimer = null; }
    this.buffer = '';
    this.lastToken = null;
    this._emit('bufferUpdate', this.buffer);
  }

  // Speaks textOverride immediately, bypassing the buffer entirely. With no
  // argument it speaks (and clears) whatever is currently buffered — the
  // "manual flush" trigger.
  speak(textOverride) {
    const bypassing = textOverride != null;
    const text = sanitizeForSpeech(bypassing ? textOverride : this.buffer);
    if (!bypassing) this.clearBuffer();
    if (text) this._synthesize(text);
  }

  // ---------------------------------------------------------------- //
  // Playback control
  // ---------------------------------------------------------------- //
  pause() {
    if (!this.isSpeaking || this.isPaused) return;
    this.isPaused = true;
    if (this._azure?.player) this._azure.player.pause();
    if (typeof window !== 'undefined' && window.speechSynthesis) window.speechSynthesis.pause();
    if (this._activeAudio) this._activeAudio.pause();
  }

  resume() {
    if (!this.isPaused) return;
    this.isPaused = false;
    if (this._azure?.player) this._azure.player.resume();
    if (typeof window !== 'undefined' && window.speechSynthesis) window.speechSynthesis.resume();
    if (this._activeAudio) this._activeAudio.play().catch(() => {});
  }

  stop() {
    const wasSpeaking = this.isSpeaking;

    if (this._azure?.synthesizer) {
      try { this._azure.synthesizer.close(); } catch { /* already closed */ }
      this._azure.synthesizer = null;
    }
    if (this._azure?.player) {
      try { this._azure.player.pause(); this._azure.player.close?.(); } catch { /* already closed */ }
      this._azure.player = null;
    }
    if (typeof window !== 'undefined' && window.speechSynthesis) window.speechSynthesis.cancel();
    this._activeUtterance = null;

    if (this._activeAudio) {
      try { this._activeAudio.pause(); this._activeAudio.currentTime = 0; } catch { /* not seekable */ }
    }
    this._releaseAudio();

    this.isPaused = false;
    if (wasSpeaking) this._signalSpeakingEnd(this._activeSynthesisSeq);

    // Cancelling the underlying engine doesn't reliably fire every tier's
    // onend/onerror (Azure's SpeakerAudioDestination in particular), which
    // would otherwise leave _synthesize()'s `await tier.run()` hanging
    // forever. Force that promise to settle now so isSpeaking/speakingEnd
    // stay consistent no matter which tier stop() interrupted.
    if (this._activeSettle) {
      const settle = this._activeSettle;
      this._activeSettle = null;
      settle();
    }
  }

  _releaseAudio() {
    if (this._activeObjectUrl) { URL.revokeObjectURL(this._activeObjectUrl); this._activeObjectUrl = null; }
    this._activeAudio = null;
  }

  setRate(speed) { this.rate = clampRate(speed); }

  setVoice(voiceName) {
    this.voiceName = resolveVoiceName(voiceName, this.voiceName);
    if (this._azure?.speechConfig) this._azure.speechConfig.speechSynthesisVoiceName = this.voiceName;
  }

  // Tears down any in-flight audio/synthesizer state and detaches all
  // listeners — call when unmounting a route or resetting detection so
  // no stale Object URL, <audio> element, or Azure synthesizer lingers.
  destroy() {
    this.stop();
    this.clearBuffer();
    this._listeners = { speakingStart: [], speakingEnd: [], bufferUpdate: [], error: [] };
    this._azure = null;
    this._azureReady = null;
    this.destroyed = true;
  }

  // ---------------------------------------------------------------- //
  // Engine tier orchestration
  // ---------------------------------------------------------------- //
  async _synthesize(text) {
    if (this.destroyed) return;
    this.stop(); // only one utterance/audio in flight per instance — forces any prior call to settle first
    const seq = ++this._synthesisSeq;
    this._activeSynthesisSeq = seq;
    this.isSpeaking = true;
    this.isPaused = false;
    // Deliberately NOT emitted here: a tier can still throw before it ever
    // produces audible audio (bad key, network down, playback rejected),
    // so onSpeakingStart is fired by each tier itself, exactly when that
    // tier's audio actually starts playing (see _markSpeakingStarted).
    this._speakingStartEmitted = false;
    this._speakingEndEmitted = false;

    const tiers = [
      { id: 1, run: () => this._speakAzure(text, seq) },
      { id: 2, run: () => this._speakGoogleTranslate(text, seq) },
      { id: 3, run: () => this._speakBrowser(text, seq) },
    ];

    let lastError = null;
    for (const tier of tiers) {
      try {
        await tier.run();
        this._signalSpeakingEnd(seq);
        return;
      } catch (err) {
        lastError = err;
        this._emit('error', { tier: tier.id, message: err?.message || String(err), error: err });
      }
    }

    // STRICT: every tier above only ever speaks az-AZ or throws — there is
    // no tier left that could speak another language, so the only correct
    // move here is to log loudly and stay silent, never substitute English/
    // Turkish/anything else.
    const message = 'All Azerbaijani (az-AZ) speech engines failed — refusing to fall back to another language.';
    console.error(`[SignLanguageTTS] ${message}`, lastError);
    this._signalSpeakingEnd(seq);
    this._emit('error', { tier: 'all', message, error: lastError });
  }

  // Called by whichever tier actually starts producing audible audio.
  // Double-guarded: `seq` stops a call that stop() has already superseded
  // (e.g. a rapid second speak() interrupting the first) from touching
  // state that now belongs to the new call, and _speakingStartEmitted
  // stops a single call from reporting more than one start.
  _markSpeakingStarted(text, seq) {
    if (seq !== this._activeSynthesisSeq || this._speakingStartEmitted) return;
    this._speakingStartEmitted = true;
    this._emit('speakingStart', text);
  }

  // Same double guard as above, for the symmetric case: stop() force-
  // settles the in-flight tier promise (emitting end itself), and that
  // same promise then "succeeds" for real via the forced settle — without
  // both guards that would double-fire, or bleed into a newer call.
  _signalSpeakingEnd(seq) {
    if (seq !== this._activeSynthesisSeq || this._speakingEndEmitted) return;
    this._speakingEndEmitted = true;
    this.isSpeaking = false;
    this._emit('speakingEnd');
  }

  // ---- Tier 1: Azure Cognitive Services Speech SDK ----
  async _ensureAzure() {
    if (this._azureReady) return this._azureReady;
    if (!this.config.azureKey || !this.config.azureRegion) {
      this._azureReady = Promise.resolve(false);
      return this._azureReady;
    }
    this._azureReady = (async () => {
      try {
        const sdk = await import('microsoft-cognitiveservices-speech-sdk');
        const speechConfig = sdk.SpeechConfig.fromSubscription(this.config.azureKey, this.config.azureRegion);
        speechConfig.speechSynthesisVoiceName = this.voiceName;
        this._azure = { sdk, speechConfig, synthesizer: null, player: null };
        return true;
      } catch (err) {
        this._emit('error', { tier: 1, stage: 'init', message: 'Azure Speech SDK unavailable', error: err });
        return false;
      }
    })();
    return this._azureReady;
  }

  async _speakAzure(text, seq) {
    const ready = await this._ensureAzure();
    if (!ready) throw new Error('azure-unavailable');

    const { sdk, speechConfig } = this._azure;
    speechConfig.speechSynthesisVoiceName = this.voiceName;

    // SpeakerAudioDestination (rather than the SDK's default output) is
    // what makes pause()/resume() possible for this tier.
    const player = new sdk.SpeakerAudioDestination();
    const audioConfig = sdk.AudioConfig.fromSpeakerOutput(player);
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig, audioConfig);
    this._azure.player = player;
    this._azure.synthesizer = synthesizer;

    const ssml = buildSsml(text, this.voiceName, this.rate);

    return new Promise((resolve, reject) => {
      this._activeSettle = () => {
        try { synthesizer.close(); } catch { /* already closed */ }
        resolve();
      };
      synthesizer.synthesisStarted = () => this._markSpeakingStarted(text, seq);
      player.onAudioEnd = () => {
        this._activeSettle = null;
        try { synthesizer.close(); } catch { /* already closed */ }
        resolve();
      };
      synthesizer.speakSsmlAsync(
        ssml,
        (result) => {
          if (result.reason === sdk.ResultReason.Canceled) {
            this._activeSettle = null;
            try { synthesizer.close(); } catch { /* already closed */ }
            reject(new Error(result.errorDetails || 'Azure synthesis canceled'));
          }
          // On success, resolution happens via player.onAudioEnd once
          // playback (not just synthesis) actually finishes.
        },
        (err) => {
          this._activeSettle = null;
          try { synthesizer.close(); } catch { /* already closed */ }
          reject(err instanceof Error ? err : new Error(String(err)));
        }
      );
    });
  }

  // ---- Tier 2: free audio stream fallback (always Azerbaijani) ----
  // Dispatches to a caller-supplied REST endpoint if configured, otherwise
  // defaults to Google Translate's public TTS stream with tl=az hardcoded
  // — the "just works, no API key" fallback that actually speaks
  // Azerbaijani instead of going silent when tier 1 isn't available.
  async _speakGoogleTranslate(text, seq) {
    if (this.config.gttsEndpoint) return this._speakCustomEndpoint(text, seq);

    const url = `https://translate.google.com/translate_tts?ie=UTF-8&q=${encodeURIComponent(text)}&tl=az&client=tw-ob`;

    return new Promise((resolve, reject) => {
      // Loaded straight as an <audio src>, not fetch()+blob: this is a
      // cross-origin, unauthenticated public endpoint with no CORS
      // headers, so fetch() would be blocked while direct media playback
      // is not.
      const audio = new Audio(url);
      audio.playbackRate = this.rate;
      this._activeAudio = audio;
      this._activeSettle = () => { this._releaseAudio(); resolve(); };
      audio.onplay = () => this._markSpeakingStarted(text, seq);
      audio.onended = () => { this._activeSettle = null; this._releaseAudio(); resolve(); };
      audio.onerror = () => {
        this._activeSettle = null;
        this._releaseAudio();
        reject(new Error('Google Translate TTS playback failed'));
      };
      audio.play().catch((err) => { this._activeSettle = null; this._releaseAudio(); reject(err); });
    });
  }

  async _speakCustomEndpoint(text, seq) {
    const url = typeof this.config.gttsEndpoint === 'function'
      ? this.config.gttsEndpoint(text, this.voiceName)
      : `${this.config.gttsEndpoint}?text=${encodeURIComponent(text)}&lang=az`;

    const response = await fetch(url);
    if (!response.ok) throw new Error(`gTTS endpoint returned ${response.status}`);
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    this._activeObjectUrl = objectUrl;

    return new Promise((resolve, reject) => {
      const audio = new Audio(objectUrl);
      audio.playbackRate = this.rate;
      this._activeAudio = audio;
      this._activeSettle = () => { this._releaseAudio(); resolve(); };
      audio.onplay = () => this._markSpeakingStarted(text, seq);
      audio.onended = () => { this._activeSettle = null; this._releaseAudio(); resolve(); };
      audio.onerror = () => {
        this._activeSettle = null;
        this._releaseAudio();
        reject(new Error('audio playback failed'));
      };
      audio.play().catch((err) => { this._activeSettle = null; this._releaseAudio(); reject(err); });
    });
  }

  // ---- Tier 3: native browser speechSynthesis, STRICT az-AZ voice only ----
  _getBrowserVoiceList() {
    return new Promise((resolve) => {
      if (typeof window === 'undefined' || !window.speechSynthesis) return resolve([]);
      const existing = window.speechSynthesis.getVoices();
      if (existing.length) return resolve(existing);
      const handler = () => {
        window.speechSynthesis.removeEventListener('voiceschanged', handler);
        resolve(window.speechSynthesis.getVoices());
      };
      window.speechSynthesis.addEventListener('voiceschanged', handler);
      // Some browsers never fire voiceschanged for an empty voice list.
      setTimeout(() => {
        window.speechSynthesis.removeEventListener('voiceschanged', handler);
        resolve(window.speechSynthesis.getVoices());
      }, 500);
    });
  }

  // Strict check, on purpose: matches voice.lang.startsWith('az') (case-
  // insensitive, so "AZ-az" style casing quirks some drivers report still
  // count), i.e. "az", "az-AZ", "az-Latn", etc. — NEVER a looser locale
  // match. If nothing in the installed voice list matches, this returns
  // null and _speakBrowser below skips speechSynthesis.speak() entirely —
  // it must NEVER be called with text.lang='az-AZ' and no matching voice,
  // since browsers silently substitute the system default voice (typically
  // en-US) in that case instead of failing, which is exactly what makes
  // Azerbaijani text come out with an English (or Turkish, etc.) accent.
  async _findAzBrowserVoice() {
    if (this._browserVoice !== undefined) return this._browserVoice;
    const voices = await this._getBrowserVoiceList();
    this._browserVoice = voices.find((v) => v.lang && /^az/i.test(v.lang)) || null;
    return this._browserVoice;
  }

  async _speakBrowser(text, seq) {
    if (typeof window === 'undefined' || !window.speechSynthesis) throw new Error('speechSynthesis-unavailable');
    const azVoice = await this._findAzBrowserVoice();
    if (!azVoice) throw new Error('no-az-browser-voice'); // CRITICAL: never fall through to speak() without one

    return new Promise((resolve, reject) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.voice = azVoice;
      utterance.lang = azVoice.lang;
      utterance.rate = this.rate;
      this._activeSettle = () => resolve();
      utterance.onstart = () => this._markSpeakingStarted(text, seq);
      utterance.onend = () => { this._activeSettle = null; resolve(); };
      utterance.onerror = (e) => {
        this._activeSettle = null;
        reject(e.error instanceof Error ? e.error : new Error('speechSynthesis error'));
      };
      this._activeUtterance = utterance;
      window.speechSynthesis.speak(utterance);
    });
  }
}

export default SignLanguageTTS;
