// Talk instead of typing, for the chat and the Studio. The words go into the message box; nothing is sent on its own.
// When Pal's server can turn speech into text (faster-whisper on this computer, or your OpenAI / Groq key), the page records with the
// browser's MediaRecorder and sends the audio to this computer's Pal only. Otherwise it falls back to the browser's own speech recognition.
// Dictation apps such as Wispr Flow need none of this: they type into the box.
(function () {
  'use strict';
  async function serverEngines() {
    try { const r = await fetch('/api/voice'); if (!r.ok) return []; return (await r.json()).engines || []; } catch (e) { return []; }
  }
  function insert(box, text, onInput) {
    if (!text) return;
    const base = box.value && !/\s$/.test(box.value) ? box.value + ' ' : box.value;
    box.value = base + text;
    box.dispatchEvent(new Event('input', { bubbles: true }));
    if (onInput) onInput();
    box.focus();
  }
  function setLive(btn, on, label) {
    btn.classList.toggle('live', on);
    btn.setAttribute('aria-pressed', String(on));
    btn.title = label;
  }
  window.palVoice = async function (btn, box, note, onInput) {
    note = note || function () {};
    const engines = await serverEngines();
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const canRecord = !!(navigator.mediaDevices && window.MediaRecorder);
    if (!(engines.length && canRecord) && !SR) {
      btn.disabled = true;
      btn.title = 'Voice input is not available here: reinstall Pal (pip install -U pal-code), or use Chrome, Edge or Safari';
      return;
    }
    let rec = null, chunks = [], stream = null, sr = null;
    const idle = engines.length && canRecord ? 'Speak instead of typing (turned into text on this computer: ' + engines[0] + ')' : 'Speak instead of typing (your browser turns it into text)';
    setLive(btn, false, idle);
    btn.addEventListener('click', async function () {
      if (rec) { rec.stop(); return; }
      if (sr) { sr.stop(); return; }
      note('');
      if (engines.length && canRecord) {
        try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
        catch (e) { note('The microphone is blocked. Allow it for this page in the browser, then try again.'); return; }
        chunks = [];
        rec = new MediaRecorder(stream);
        rec.ondataavailable = function (e) { if (e.data && e.data.size) chunks.push(e.data); };
        rec.onstop = async function () {
          stream.getTracks().forEach(function (t) { t.stop(); });
          const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
          rec = null;
          setLive(btn, false, 'Turning speech into text…');
          btn.disabled = true;
          try {
            const r = await fetch('/api/transcribe', { method: 'POST', headers: { 'Content-Type': blob.type.split(';')[0] || 'audio/webm' }, body: blob });
            const d = await r.json();
            if (!r.ok) note(d.error || 'Speech-to-text failed'); else if (!d.text) note('I did not hear anything.'); else insert(box, d.text, onInput);
          } catch (e) { note('Speech-to-text failed: ' + e.message); }
          btn.disabled = false;
          setLive(btn, false, idle);
        };
        rec.start();
        setLive(btn, true, 'Listening… click again to stop');
        return;
      }
      sr = new SR();
      sr.interimResults = true; sr.continuous = true; sr.lang = navigator.language || 'en-US';
      const start = box.value && !/\s$/.test(box.value) ? box.value + ' ' : box.value;
      sr.onresult = function (e) {                     // the words so far, shown as they are heard
        let t = ''; for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
        box.value = start + t; box.dispatchEvent(new Event('input', { bubbles: true })); if (onInput) onInput();
      };
      sr.onerror = function (e) { note(e.error === 'not-allowed' || e.error === 'service-not-allowed' ? 'The microphone is blocked. Allow it for this page in the browser, then try again.' : e.error === 'no-speech' ? 'I did not hear anything.' : 'Voice input stopped: ' + e.error); };
      sr.onend = function () { sr = null; setLive(btn, false, idle); };
      try { sr.start(); setLive(btn, true, 'Listening… click again to stop'); } catch (e) { sr = null; }
    });
  };
})();
