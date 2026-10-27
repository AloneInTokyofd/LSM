'use strict';
const $ = id => document.getElementById(id);
let state, language = 'lsm', mode = 'test', session = null, stream = null;
let generation = 0, timer = null, countdown = false, recording = false, frameReady = false;
let history = [], lastSpoken = null, checks = 0, correct = 0, lastJobState = '';
let stateRequest = null;
let selectedDraft = null, displayedDraft = null, savingDraft = false;
let selectedLetter = null, addingAlphabet = false;
const batch = new ContinuousCapture();
const rigOverlay = new RigOverlay($('rig-overlay'));
let batchPause = 3;
let cameraStarting = false, cameraListing = false;
let deletingRecordings = false, adminLanguage = null, displayedModelKey = null;
const profile = () => state?.profiles.find(p => p.id === language);

function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').className = error ? 'error' : '';
  $('notice').hidden = !message;
}
async function api(path, body) {
  const options = body === undefined ? {} : {method: 'POST', headers: {'Content-Type':'application/json', 'X-LSM-Token':state?.token || ''}, body:JSON.stringify(body)};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 404 && path.startsWith('/api/admin/')) {
      throw new Error('El servidor activo es una versión anterior sin administración. Detén web_app.py con Ctrl+C, vuelve a ejecutarlo y recarga la página.');
    }
    throw new Error(data.error || 'No se pudo completar la solicitud.');
  }
  return data;
}
function makeOption(value, text) {
  const element = document.createElement('option'); element.value = value; element.textContent = text; return element;
}
function render() {
  const p = profile();
  if (!p) return;
  const modelKey = `${language}:${p.model?.created_at || 'disabled'}`;
  if (displayedModelKey !== modelKey) { displayedModelKey = modelKey; clearHistory(); }
  $('languages').replaceChildren();
  for (const item of state.profiles) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'language-button'; button.setAttribute('aria-current', String(item.id === language));
    const code = document.createElement('span'); code.className = 'lang-code'; code.textContent = item.code;
    const copy = document.createElement('span'); copy.className = 'lang-copy'; copy.textContent = item.written;
    const region = document.createElement('small'); region.textContent = item.region; copy.append(region);
    const dot = document.createElement('span'); dot.className = 'ready-dot' + (item.model ? ' ready' : ''); dot.title = item.model ? 'Modelo entrenado' : 'Sin modelo';
    button.append(code, copy, dot); button.addEventListener('click', () => selectLanguage(item.id)); $('languages').append(button);
  }
  $('breadcrumb').textContent = p.code;
  $('language-name').textContent = p.name;
  $('language-detail').textContent = `${p.region} · Texto de salida en ${p.written.toLowerCase()}`;
  $('output-language').textContent = p.written.toUpperCase();
  $('model-badge').textContent = p.model ? 'Modelo entrenado' : (p.model_error ? 'Modelo no disponible' : 'Necesita entrenamiento');
  $('model-badge').className = 'badge' + (p.model ? ' ready' : '');
  $('legacy-note').hidden = !state.legacy_model;
  const drafts = p.drafts || [];
  if (!drafts.some(d => d.id === selectedDraft)) selectedDraft = drafts[0]?.id || null;
  const draftKey = `${language}:${selectedDraft || ''}`;
  if (draftKey !== displayedDraft) { $('sign-name').value = selectedLetter || ''; displayedDraft = draftKey; }
  $('name-capture').hidden = !selectedDraft;
  $('draft-language').textContent = `Esta grabación pertenece a ${p.code} · ${p.name}. Se conservará pendiente hasta que la guardes o descartes.`;
  $('draft-picker').replaceChildren(...drafts.map(d => makeOption(d.id, new Date(d.created_at * 1000).toLocaleString())));
  if (selectedDraft) $('draft-picker').value = selectedDraft;
  $('draft-picker').hidden = drafts.length < 2;
  $('draft-picker-label').hidden = drafts.length < 2;
  $('sign-suggestions').replaceChildren(...p.labels.map(item => makeOption(item.text, item.text)));
  const expected = $('expected-label').value;
  $('expected-label').replaceChildren(makeOption('', 'Práctica libre'), ...(p.model?.labels || []).map(item => makeOption(item.id, item.text)));
  if (p.model?.labels.some(x => x.id === expected)) $('expected-label').value = expected;
  $('sample-list').replaceChildren();
  for (const item of p.labels) {
    const row = document.createElement('div'); row.className = 'sample-row' + (item.samples >= state.minimum_samples ? ' enough' : '');
    const label = document.createElement('span'); label.textContent = item.text;
    const count = document.createElement('strong'); count.textContent = `${item.samples} / ${state.minimum_samples} mín. · recomendado 30–${state.recommended_samples}`;
    if (item.samples > 0 && !p.model?.labels.some(x => x.id === item.id)) {
      const status = document.createElement('small'); status.className = 'pending-label'; status.textContent = 'Por entrenar'; label.append(status);
    }
    row.append(label, count);
    if (item.invalid) { const invalid = document.createElement('small'); invalid.textContent = ` ${item.invalid} inválidos`; row.append(invalid); }
    $('sample-list').append(row);
  }
  const populated = p.labels.filter(x => x.samples > 0);
  const eligible = populated.length >= 2 && populated.every(x => x.samples >= state.minimum_samples);
  const busy = state.job?.state === 'running';
  renderAdmin();
  $('train-model').disabled = !eligible || busy || recording || countdown || batch.active || !!selectedDraft || savingDraft;
  $('train-reason').textContent = selectedDraft ? 'Pon nombre y guarda tu grabación pendiente antes de entrenar.' : busy ? 'Hay un entrenamiento en curso. Puedes seguir viendo su avance.' : eligible ? `Se entrenarán ${populated.length} señas de ${p.code}. Recomendamos entre 30 y ${state.recommended_samples} ejemplos variados por palabra o letra. Las tandas pueden ser de 5 tomas; el entrenamiento requiere 30 por palabra o letra.` : `Completa al menos ${state.minimum_samples} ejemplos de cada seña que empezaste y reúne 2 señas diferentes. Recomendado: 30–${state.recommended_samples} por palabra o letra.`;
  $('start-camera').disabled = busy || cameraStarting;
  $('toggle-rig').disabled = !stream || cameraStarting;
  $('camera-device').disabled = busy || cameraStarting || savingDraft;
  $('refresh-cameras').disabled = cameraListing || cameraStarting;
  $('record').disabled = !stream || !frameReady || countdown || recording || batch.active || busy || !!selectedDraft || savingDraft;
  $('save-capture').disabled = savingDraft || busy || batch.active;
  $('discard-capture').disabled = savingDraft || batch.active;
  $('sign-name').disabled = savingDraft || busy || batch.active;
  $('draft-picker').disabled = batch.active;
  $('start-batch').disabled = !stream || !frameReady || countdown || recording || batch.active || savingDraft || busy;
  $('stop-batch').disabled = !batch.active;
  $('batch-limit').disabled = batch.active;
  $('batch-pause').disabled = batch.active;
  $('batch-status').textContent = batch.active ? `Tanda en curso: ${batch.count} / ${batch.limit} tomas completas. Ponles nombre al terminar o pulsa “Detener tanda”.` : batch.count ? `Tanda detenida: ${batch.count} tomas completas conservadas. Nombra y guarda cada una usando “Grabaciones pendientes”.` : 'Activa la cámara para iniciar. No necesitas pulsar “Grabar una seña” entre tomas.';
  $('expected-label').disabled = !p.model;
  if (!stream && !history.length) resetResult();
  const summary = $('model-summary'); summary.replaceChildren();
  if (p.model) {
    const strong = document.createElement('strong'); strong.textContent = `${p.code}: ${p.model.labels.length} señas entrenadas. `;
    summary.append(strong, `Aciertos en prueba reservada: ${(100 * p.model.test_accuracy).toFixed(1)}% (${p.model.split.test} ejemplos). ${p.model.evaluation_note}`);
  } else {
    summary.textContent = p.model_error || `Este perfil está preparado para aprender. Todavía no tiene un modelo de ${p.code}; activar la cámara no generará detecciones inventadas.`;
  }
  $('model-metrics').replaceChildren();
  if (p.model?.report) {
    const title = document.createElement('h3'); title.textContent = 'Resultados por seña en la prueba reservada'; $('model-metrics').append(title);
    for (const label of p.model.labels) {
      const metric = p.model.report[label.id] || p.model.report[label.text];
      if (!metric) continue;
      const row = document.createElement('p');
      row.textContent = `${label.text}: precisión ${(100 * metric.precision).toFixed(1)}% · recuperación ${(100 * metric.recall).toFixed(1)}% · F1 ${(100 * metric['f1-score']).toFixed(1)}% · ${metric.support} ejemplos`;
      $('model-metrics').append(row);
    }
    const decision = p.model.test_decisions;
    const comparison = p.model.model_comparison;
    if (comparison?.previous_validation) {
      const row = document.createElement('p');
      row.textContent = `Comparación en validación: modelo anterior ${(100 * comparison.previous_validation.accuracy).toFixed(1)}% de aciertos; versión actual ${(100 * comparison.candidate_validation.accuracy).toFixed(1)}%. ${comparison.note}`;
      $('model-metrics').append(row);
    }
    if (decision) {
      const row = document.createElement('p');
      row.textContent = `Predicciones aceptadas con el filtro: ${decision.accepted} de ${decision.evaluated}. Aciertos entre las aceptadas: ${decision.accepted_accuracy === null ? 'sin predicciones aceptadas' : (100 * decision.accepted_accuracy).toFixed(1) + '%'}. Las demás se rechazan como dudosas.`;
      $('model-metrics').append(row);
    }
    if (!p.model.preprocessing) {
      const row = document.createElement('p'); row.textContent = 'Modelo anterior: reúne al menos 30 ejemplos por palabra o letra (recomendado: 30–60) y reentrena para aplicar la normalización y el entrenamiento mejorado.'; $('model-metrics').append(row);
    }
  }
  renderAlphabet();
  renderJob();
}
function renderAlphabet() {
  const p = profile(), alphabet = p.alphabet;
  if (!alphabet) return;
  $('alphabet-title').textContent = `${alphabet.title} · ${p.code}`;
  $('alphabet-note').textContent = alphabet.note;
  $('alphabet-count').textContent = `${alphabet.symbols.length} entradas básicas`;
  const registered = alphabet.symbols.filter(x => x.label_id).length;
  const trained = alphabet.symbols.filter(x => x.trained).length;
  $('alphabet-progress').textContent = `${registered} / ${alphabet.symbols.length} añadidas al vocabulario · ${trained} incluidas en el modelo. Pulsa una letra para grabar sus ejemplos.`;
  const busy = addingAlphabet || batch.active || state.job?.state === 'running';
  $('add-alphabet').disabled = busy || registered === alphabet.symbols.length;
  $('add-alphabet').textContent = registered === alphabet.symbols.length ? 'Abecedario añadido' : 'Añadir todo el abecedario';
  $('add-all-alphabets').disabled = busy;
  $('add-alphabet-extras').disabled = busy || alphabet.extras.every(x => x.label_id);
  $('alphabet-extras').hidden = !alphabet.extras.length;
  const buildGrid = (id, entries) => {
    $(id).replaceChildren();
    for (const entry of entries) {
      const button = document.createElement('button'); button.type = 'button';
      button.className = 'letter-tile' + (entry.trained ? ' trained' : '');
      button.setAttribute('aria-pressed', String(selectedLetter === entry.symbol));
      button.setAttribute('aria-label', `${entry.symbol}: ${entry.samples} ejemplos${entry.trained ? ', incluida en el modelo' : ''}`);
      button.disabled = recording || countdown || savingDraft || batch.active;
      const symbol = document.createElement('span'); symbol.textContent = entry.symbol;
      const count = document.createElement('small'); count.textContent = `${entry.samples} / ${state.minimum_samples}`;
      button.append(symbol, count);
      button.addEventListener('click', () => {
        selectedLetter = entry.symbol;
        if (selectedDraft) $('sign-name').value = entry.symbol;
        renderAlphabet();
        if (selectedDraft) { $('name-capture').scrollIntoView({behavior:'smooth',block:'center'}); $('sign-name').focus({preventScroll:true}); }
        else $('camera-workspace').scrollIntoView({behavior:'smooth',block:'start'});
      });
      $(id).append(button);
    }
  };
  buildGrid('alphabet-grid', alphabet.symbols); buildGrid('alphabet-extra-grid', alphabet.extras);
  $('alphabet-selection').textContent = selectedLetter ? `Letra seleccionada: ${selectedLetter} · ${p.code}. Al terminar la captura, confirma ese nombre y guarda el ejemplo.` : 'Selecciona una letra para preparar su nombre al terminar la grabación.';
  $('clear-alphabet-selection').hidden = !selectedLetter;
  $('alphabet-source').hidden = !alphabet.source;
  if (alphabet.source) $('alphabet-source').href = alphabet.source;
}
async function addAlphabet(scope, extras = false) {
  if (addingAlphabet) return;
  const chosenLanguage = language;
  addingAlphabet = true; renderAlphabet();
  try {
    const result = await api('/api/alphabet', {language:chosenLanguage, scope, extras});
    await refresh();
    const added = result.results ? result.results.reduce((total, item) => total + item.added, 0) : result.added;
    notice(`${added} letras o caracteres añadidos${scope === 'all' ? ' a los vocabularios de las siete lenguas' : ' al vocabulario seleccionado'}. Ahora puedes grabar ejemplos y entrenarlos.`);
  } catch (error) { notice(error.message, true); }
  finally { addingAlphabet = false; renderAlphabet(); }
}

function renderJob() {
  const job = state.job; $('job-panel').hidden = !job;
  if (!job) return;
  const code = state.profiles.find(p => p.id === job.language)?.code || '';
  $('job-title').textContent = job.state === 'completed' ? (job.activated === false ? `${code} · Se conservó el modelo anterior` : `${code} · Modelo listo para probar`) : job.state === 'failed' ? `${code} · El entrenamiento no se completó` : `${code} · Época ${job.epoch || 0} de ${job.epochs || '…'}`;
  $('job-description').textContent = job.message || '';
  $('job-progress').max = job.epochs || 40; $('job-progress').value = job.state === 'completed' ? (job.epochs || 40) : (job.epoch || 0);
  $('job-metrics').textContent = job.result ? `Prueba reservada: ${(100 * job.result.test_accuracy).toFixed(1)}% de aciertos` : job.metrics ? `Validación: ${(100 * (job.metrics.val_accuracy || 0)).toFixed(1)}% · Pérdida: ${(job.metrics.val_loss || 0).toFixed(3)}` : 'Preparando el modelo…';
  const fingerprint = `${job.id}:${job.state}`;
  if (fingerprint !== lastJobState && job.state === 'completed') notice(job.activated === false ? job.message : 'Entrenamiento terminado. Abre “Probar modelo” y activa la cámara para comprobarlo.');
  if (fingerprint !== lastJobState && job.state === 'failed') notice(job.message, true);
  lastJobState = fingerprint;
}
async function refresh() {
  if (stateRequest) return stateRequest;
  stateRequest = api('/api/state').then(data => { const changed = JSON.stringify(data) !== JSON.stringify(state); state = data; if (changed) render(); }).finally(() => { stateRequest = null; });
  return stateRequest;
}
async function selectLanguage(id) {
  if (id === language) return;
  await stopCamera(); language = id; selectedLetter = null; batch.count = 0; clearHistory(); notice(''); render();
}
async function selectMode(next) {
  if (next === mode) return;
  await stopCamera(); mode = next;
  for (const button of document.querySelectorAll('[data-mode]')) button.setAttribute('aria-selected', String(button.dataset.mode === mode));
  $('alphabet-panel').hidden = mode !== 'train';
  $('camera-workspace').hidden = mode === 'guide'; $('guide').hidden = mode !== 'guide';
  $('capture-controls').hidden = mode !== 'train'; $('training-results').hidden = mode !== 'train';
  $('test-controls').hidden = mode !== 'test'; $('test-results').hidden = mode !== 'test';
  notice(''); render();
}
function resetResult() {
  $('result-word').textContent = profile()?.model ? 'Listo para practicar.' : 'Todo empieza con un gesto.';
  $('result-description').textContent = profile()?.model ? 'Activa la cámara y realiza una de las señas entrenadas.' : 'Primero necesitas un modelo entrenado para esta lengua.';
  $('confidence').textContent = '—'; $('confidence-progress').value = 0; $('stability').textContent = 'Esperando una secuencia';
}
function renderAdmin() {
  const p = profile();
  const previous = adminLanguage === language ? $('admin-label').value : '';
  if (adminLanguage !== language) {
    $('admin-password').value = ''; $('admin-confirm').checked = false; $('admin-message').textContent = '';
  }
  adminLanguage = language;
  const available = p.labels.filter(item => item.samples + item.invalid > 0);
  $('admin-label').replaceChildren(makeOption('', 'Selecciona una palabra o letra'),
    ...available.map(item => makeOption(item.id, `${item.text} · ${item.samples + item.invalid} grabaciones`)));
  if (available.some(item => item.id === previous)) $('admin-label').value = previous;
  else $('admin-confirm').checked = false;
  const selected = available.find(item => item.id === $('admin-label').value);
  $('admin-target').textContent = selected ? `${p.code} · ${selected.text}: se retirarán ${selected.samples + selected.invalid} grabaciones del conjunto de entrenamiento.` : `${p.code} · Elige la palabra o letra cuyos ejemplos quieres borrar.`;
  const blocked = deletingRecordings || state.job?.state === 'running' || recording || countdown || batch.active;
  $('admin-label').disabled = blocked;
  $('admin-password').disabled = blocked;
  $('admin-confirm').disabled = blocked;
  $('admin-delete').disabled = blocked || !selected;
}
async function deleteRecordings(event) {
  event.preventDefault();
  if (deletingRecordings) return;
  const selectedLanguage = language, label = $('admin-label').value;
  const password = $('admin-password').value, confirmed = $('admin-confirm').checked;
  if (!label || !password || !confirmed) return;
  deletingRecordings = true;
  $('admin-password').value = '';
  renderAdmin();
  try {
    await stopCamera();
    const result = await api('/api/admin/delete-recordings', {language:selectedLanguage, label, password, confirmed});
    if (language === selectedLanguage) {
      $('admin-confirm').checked = false;
      $('admin-message').textContent = `Se retiraron ${result.deleted} grabaciones. Se conservaron en la carpeta de recuperación. ${result.model_invalidated ? 'El modelo anterior se desactivó para que no siga reconociendo la seña borrada. Reentrena con los ejemplos restantes.' : 'Los ejemplos de esta seña no pertenecían al modelo activo.'}`;
    }
    await refresh();
  } catch (error) {
    if (language === selectedLanguage) $('admin-message').textContent = error.message;
  } finally {
    deletingRecordings = false;
    render();
  }
}
async function refreshCameras() {
  if (cameraListing) return;
  cameraListing = true;
  $('refresh-cameras').disabled = true;
  try {
    if (!navigator.mediaDevices?.enumerateDevices) throw new Error('Este navegador no permite consultar las cámaras. Usa la dirección local y permite el acceso a la cámara.');
    const selected = $('camera-device').value;
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === 'videoinput');
    const active = stream?.getVideoTracks()[0]?.getSettings().deviceId;
    $('camera-device').replaceChildren(makeOption('', 'Cámara predeterminada'),
      ...devices.map((device, index) => makeOption(device.deviceId, device.label || `Cámara ${index + 1}`)));
    const choice = active || selected;
    if (devices.some(device => device.deviceId === choice)) $('camera-device').value = choice;
    $('camera-device-status').textContent = devices.length
      ? `${devices.length} cámara(s) disponible(s). ${devices.some(device => !device.label) ? 'Activa la cámara para mostrar sus nombres.' : 'Selecciona la que quieras usar.'}`
      : 'No hay cámaras visibles para el navegador. Conecta una cámara y permite el acceso.';
    if (active && !devices.some(device => device.deviceId === active)) {
      await stopCamera();
      notice('La cámara activa se desconectó. Selecciona otra y actívala.', true);
    }
  } catch (error) {
    $('camera-device-status').textContent = error.message;
  } finally {
    cameraListing = false;
    $('refresh-cameras').disabled = cameraStarting;
  }
}
async function startCamera() {
  if (stream || cameraStarting) return;
  cameraStarting = true;
  $('start-camera').disabled = true;
  $('camera-device').disabled = true;
  const mark = ++generation;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Abre esta página en http://127.0.0.1:8765 con un navegador que permita cámara.');
    const deviceId = $('camera-device').value;
    const acquired = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},
      ...(deviceId ? {deviceId:{exact:deviceId}} : {facingMode:'user'})},audio:false});
    if (mark !== generation) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired;
    acquired.getVideoTracks()[0].addEventListener('ended', () => {
      if (stream === acquired) { stopCamera(); notice('La cámara se desconectó. Selecciona otra cámara.', true); }
    }, {once:true});
    const created = await api('/api/session', {language});
    if (mark !== generation) { await api('/api/close', {session:created.session}); return; }
    session = created.session;
    $('camera').srcObject = stream; await $('camera').play();
    $('camera').hidden = false; $('camera-placeholder').hidden = true; $('stop-camera').disabled = false;
    $('camera-state').textContent = 'Preparando detección…';
    $('capture-instruction').textContent = mode === 'train' ? 'Pulsa “Grabar una seña”. Le pondrás nombre al terminar.' : profile().model ? 'Realiza una seña dentro del encuadre.' : 'Vista previa · Esta lengua todavía no tiene modelo.';
    notice(''); sendFrame(mark); await refreshCameras();
  } catch (error) {
    await stopCamera();
    notice(error.name === 'NotAllowedError' ? 'Permite la cámara para este sitio y para tu navegador en la configuración de privacidad de macOS.' : error.name === 'OverconstrainedError' || error.name === 'NotFoundError' ? 'La cámara seleccionada no está disponible. Pulsa Detectar cámaras y elige otra.' : error.message, true);
  } finally { cameraStarting = false; render(); }
}
async function stopCamera() {
  rigOverlay.clear();
  batch.stop();
  ++generation; clearTimeout(timer); countdown = false; recording = false; frameReady = false;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null; $('camera').srcObject = null; $('camera').hidden = true; $('camera-placeholder').hidden = false;
  $('stop-camera').disabled = true; $('record').disabled = true; $('record').textContent = 'Grabar una seña';
  $('camera-state').textContent = 'Cámara apagada'; $('capture-instruction').textContent = 'La cámara se activa solo cuando tú lo decides.';
  $('frame-progress').value = 0; $('frame-counter').textContent = '0 / 30'; $('hands-status').textContent = 'Tu mano derecha aparecerá a la derecha.';
  const previous = session; session = null;
  if (previous) { try { await api('/api/close', {session:previous}); } catch (_) {} }
  if (state) render();
}
async function sendFrame(mark) {
  if (!stream || mark !== generation || !session) return;
  const started = performance.now();
  try {
    const canvas = $('frame-canvas'), video = $('camera');
    const width = Math.min(video.videoWidth || 640, 640);
    canvas.width = width; canvas.height = Math.round(width * (video.videoHeight || 480) / (video.videoWidth || 640));
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const result = await api('/api/frame', {session, image:canvas.toDataURL('image/jpeg', .7).split(',')[1], mode, rig:rigOverlay.enabled});
    if (mark !== generation) return;
    rigOverlay.draw(result.rig);
    if (!frameReady) { frameReady = true; render(); }
    $('camera-state').textContent = result.hands ? 'Manos visibles' : 'Buscando manos';
    $('hands-status').textContent = result.hands ? 'Manos detectadas · Mantén el movimiento natural.' : 'Coloca ambas manos dentro del encuadre.';
    const count = recording ? result.captured : result.buffer;
    $('frame-progress').max = state.sequence_length; $('frame-progress').value = count;
    $('frame-counter').textContent = `${count} / ${state.sequence_length}`;
    if (result.draft_id) {
      selectedDraft = result.draft_id;
      recording = false;
      if (batch.active) {
        const serial = batch.serial;
        const progress = batch.complete(result.draft_id, serial);
        if (progress && !progress.done && batch.current(serial)) {
          await refresh();
          if (mark !== generation) return;
          notice(`Toma ${progress.count} guardada. Prepárate: la siguiente empieza en ${batchPause} segundos.`);
          void recordExample(true, serial);
        } else {
          await stopCamera();
          await refresh();
          notice(`Tanda terminada: ${batch.count} tomas completas. Nombra y guarda cada una en “Grabaciones pendientes”.`);
          $('name-capture').scrollIntoView({behavior:'smooth', block:'center'});
          $('sign-name').focus({preventScroll:true});
          return;
        }
      } else {
      await stopCamera();
      await refresh();
      notice('Grabación completa. Escribe qué significa y pulsa “Guardar e integrar”.');
      $('name-capture').scrollIntoView({behavior:'smooth', block:'center'});
      $('sign-name').focus({preventScroll:true});
      return;
      }
    }
    if (mode === 'test') showPrediction(result);
  } catch (error) {
    if (mark !== generation) return;
    notice(error.message, true);
    // Cerrar la sesión también descarta cualquier captura parcial en el servidor.
    await stopCamera(); return;
  }
  if (mark === generation) timer = setTimeout(() => sendFrame(mark), Math.max(0, 1000 / state.target_fps - (performance.now() - started)));
}
function showPrediction(result) {
  if (!profile().model) return;
  $('confidence').textContent = result.candidate ? `${(100 * result.confidence).toFixed(1)}%` : '—';
  $('confidence-progress').value = 100 * result.confidence;
  $('stability').textContent = `${result.stability} / 5 confirmaciones estables`;
  if (result.candidate) {
    $('result-word').textContent = result.candidate.text;
    $('result-description').textContent = result.rejection_reason || (result.confirmed ? 'Seña confirmada y añadida a tu sesión.' : 'Candidata del modelo. Esperando confianza y estabilidad suficientes.');
  } else {
    $('result-word').textContent = result.hands ? 'Observando el movimiento…' : 'Esperando tus manos…';
    $('result-description').textContent = 'Completa una secuencia con una de las señas entrenadas.';
  }
  if (result.confirmed) {
    const expected = $('expected-label').value;
    if (expected) { checks++; if (expected === result.confirmed.id) correct++; }
    history.push({...result.confirmed, confidence:result.confidence, expected, time:new Date().toISOString()});
    lastSpoken = result.confirmed.text;
    $('speak').disabled = !('speechSynthesis' in window);
    renderHistory();
  }
}
function renderHistory() {
  $('history').replaceChildren();
  for (const item of history.slice(-12)) { const chip = document.createElement('span'); chip.className = 'chip'; chip.textContent = item.text; $('history').append(chip); }
  $('history-empty').hidden = history.length > 0;
  $('practice-score').textContent = `${correct} / ${checks} aciertos`;
  $('download').disabled = !history.length;
}
function clearHistory() {
  history = []; checks = correct = 0; lastSpoken = null; $('speak').disabled = true;
  renderHistory(); resetResult();
}
async function recordExample(automatic = false, serial = null) {
  if (!stream || !frameReady || recording || countdown || (automatic ? !batch.current(serial) : (selectedDraft || batch.active))) return;
  countdown = true; render();
  const mark = generation;
  try {
    for (let n = automatic ? batchPause : 3; n > 0; n--) {
      if (mark !== generation || (automatic && !batch.current(serial))) return;
      $('record').textContent = `Prepárate… ${n}`; $('capture-instruction').textContent = `La grabación empieza en ${n}…`;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    if (mark !== generation || (automatic && !batch.current(serial))) return;
    await api('/api/capture', {session, consecutive: automatic});
    if (mark !== generation || (automatic && !batch.current(serial))) return;
    recording = true; $('record').textContent = 'Grabando…'; $('capture-instruction').textContent = 'Haz la seña completa ahora.';
  } catch (error) { notice(error.message, true); if (automatic) await stopCamera(); }
  finally { if (mark === generation) { countdown = false; render(); } }
}
async function startContinuous() {
  if (!stream || !frameReady || recording || countdown || batch.active || savingDraft) return;
  try {
    const pause = Number($('batch-pause').value);
    if (!Number.isInteger(pause) || pause < 1 || pause > 10) throw new Error('Elige una pausa de 1 a 10 segundos.');
    batchPause = pause;
    const serial = batch.start(Number($('batch-limit').value));
    notice('Tanda iniciada. Repite la seña después de cada cuenta atrás; podrás nombrar las tomas al terminar.');
    await recordExample(true, serial);
  } catch (error) { notice(error.message, true); }
  finally { render(); }
}
async function stopContinuous() {
  await stopCamera();
  await refresh();
  notice('Tanda detenida. Las tomas completas están conservadas; la toma incompleta se descartó. Puedes nombrarlas y guardarlas ahora.');
}
async function saveNamedCapture(event) {
  event.preventDefault();
  if (!selectedDraft || savingDraft) return;
  const name = $('sign-name').value.trim();
  if (!name) return notice('Escribe el nombre o significado de la seña.', true);
  const capturedLanguage = language, draft = selectedDraft;
  savingDraft = true; render();
  try {
    const result = await api('/api/capture/save', {language:capturedLanguage, draft_id:draft, name});
    selectedDraft = null;
    await refresh();
    if (language === capturedLanguage) notice(`“${result.label.text}” se integró al vocabulario de ${profile().code}. Puedes grabar más ejemplos con ese nombre y actualizar el modelo al completar los requisitos.`);
  } catch (error) { notice(error.message, true); }
  finally { savingDraft = false; render(); }
}
async function discardCapture() {
  if (!selectedDraft || savingDraft) return;
  savingDraft = true; render();
  try {
    await api('/api/capture/discard', {language, draft_id:selectedDraft});
    selectedDraft = null; await refresh(); notice('Grabación descartada. Ya puedes activar la cámara y repetirla.');
  } catch (error) { notice(error.message, true); }
  finally { savingDraft = false; render(); }
}

async function trainModel() {
  const epochs = Number($('epochs').value);
  if (!Number.isInteger(epochs) || epochs < 5 || epochs > 200) return notice('Elige entre 5 y 200 épocas.', true);
  $('train-model').disabled = true;
  try {
    await stopCamera(); await api('/api/train', {language, epochs});
    notice('Entrenamiento iniciado. El progreso aparecerá debajo; puedes dejar esta página abierta.'); await refresh();
  } catch (error) { notice(error.message, true); render(); }
}
function speak() {
  if (!lastSpoken || !window.speechSynthesis) return;
  const locale = profile().locale;
  const voices = speechSynthesis.getVoices();
  const voice = voices.find(v => v.lang === locale) || voices.find(v => v.lang.split('-')[0] === locale.split('-')[0]);
  if (!voice) return notice(`Tu navegador no tiene una voz de ${profile().written.toLowerCase()} disponible. El texto sigue disponible.`, true);
  speechSynthesis.cancel(); const speech = new SpeechSynthesisUtterance(lastSpoken); speech.lang = locale; speech.voice = voice; speechSynthesis.speak(speech);
}
$('start-camera').addEventListener('click', startCamera);
$('toggle-rig').addEventListener('click', () => {
  rigOverlay.setEnabled(!rigOverlay.enabled);
  $('toggle-rig').setAttribute('aria-pressed', String(rigOverlay.enabled));
  $('toggle-rig').textContent = rigOverlay.enabled ? 'Ocultar rig' : 'Activar rig';
});
$('admin-delete-form').addEventListener('submit', deleteRecordings);
$('admin-label').addEventListener('change', () => { $('admin-confirm').checked = false; $('admin-message').textContent = ''; renderAdmin(); });
$('refresh-cameras').addEventListener('click', refreshCameras);
$('camera-device').addEventListener('change', async () => {
   if (stream) { await stopCamera(); await startCamera(); }
 });
if (navigator.mediaDevices?.addEventListener) navigator.mediaDevices.addEventListener('devicechange', refreshCameras);
refreshCameras();
$('stop-camera').addEventListener('click', stopCamera);
$('record').addEventListener('click', () => recordExample());
$('start-batch').addEventListener('click', startContinuous);
$('stop-batch').addEventListener('click', stopContinuous);
$('add-alphabet').addEventListener('click', () => addAlphabet('language'));
$('add-all-alphabets').addEventListener('click', () => addAlphabet('all'));
$('add-alphabet-extras').addEventListener('click', () => addAlphabet('language', true));
$('clear-alphabet-selection').addEventListener('click', () => { selectedLetter = null; renderAlphabet(); });
$('name-capture').addEventListener('submit', saveNamedCapture);
$('discard-capture').addEventListener('click', discardCapture);
$('draft-picker').addEventListener('change', () => { selectedDraft = $('draft-picker').value; render(); });
$('train-model').addEventListener('click', trainModel);
$('clear-history').addEventListener('click', async () => {
  clearHistory(); if (session) { try { await api('/api/reset', {session}); } catch (e) { notice(e.message, true); } }
});
$('speak').addEventListener('click', speak);
$('download').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({language, name:profile().name, records:history, checked:checks, correct}, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = `sesion-${language}-${Date.now()}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
for (const tab of document.querySelectorAll('[data-mode]')) tab.addEventListener('click', () => selectMode(tab.dataset.mode));
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) stopCamera(); });
window.addEventListener('pagehide', () => { if (stream) stream.getTracks().forEach(t => t.stop()); });
refresh().catch(error => notice(`No se pudo conectar con el servidor local: ${error.message}`, true));
setInterval(() => refresh().catch(error => notice(`No se pudo actualizar el estado: ${error.message}`, true)), 4000);
