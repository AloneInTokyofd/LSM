'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const app = fs.readFileSync(require('node:path').join(__dirname, '../static/app.js'), 'utf8');
const cameraCode = app.slice(app.indexOf('async function refreshCameras()'), app.indexOf('async function sendFrame('));
const elements = new Map();
const $ = id => {
  if (!elements.has(id)) elements.set(id, {value:'', textContent:'', disabled:false, hidden:false, play:async () => {}, replaceChildren(...options) { this.options = options; this.value = ''; }});
  return elements.get(id);
};
let devices = [{kind:'videoinput', deviceId:'builtin', label:'FaceTime'}, {kind:'videoinput', deviceId:'usb', label:'USB'}, {kind:'audioinput', deviceId:'mic'}];
let stopped = 0, constraints, ended, messages = [], closed = 0, sent = 0;
const streamMock = () => ({getTracks:() => [{stop:() => stopped++}], getVideoTracks:() => [{getSettings:() => ({deviceId:'usb'}), addEventListener:(event, callback) => { ended = callback; }}]});
const sandbox = { $, navigator:{mediaDevices:{enumerateDevices:async () => devices, getUserMedia:async c => { constraints = c; return streamMock(); }}},
  makeOption:(value, text) => ({value, text}), api:async path => { if (path === '/api/close') closed++; return {session:'session-id'}; },
  render:() => {}, notice:(text,error) => messages.push({text,error}), sendFrame:() => sent++, clearTimeout, console,
  profile:() => ({model:true}), batch:{stop:() => {}}, language:'lsm', mode:'train', state:{} };
vm.createContext(sandbox);
vm.runInContext('let stream=null, session=null, generation=0, timer=null, countdown=false, recording=false, frameReady=false, cameraStarting=false, cameraListing=false;'+cameraCode, sandbox);
(async () => {
  await sandbox.refreshCameras();
  assert.deepEqual($('camera-device').options.map(x => x.value), ['', 'builtin', 'usb']);
  $('camera-device').value = 'usb';
  await sandbox.startCamera();
  assert.equal(constraints.video.deviceId.exact, 'usb');
  assert.equal(constraints.audio, false);
  assert.equal($('camera-device').value, 'usb');
  assert.equal(sent, 1);
  await sandbox.stopCamera();
  assert.equal(stopped, 1); assert.equal(closed, 1);
  // Preserve selection on refresh, remove it if disconnected.
  await sandbox.refreshCameras(); assert.equal($('camera-device').value, 'usb');
  devices = devices.filter(x => x.deviceId !== 'usb');
  await sandbox.refreshCameras(); assert.equal($('camera-device').value, '');
  sandbox.navigator.mediaDevices.getUserMedia = async () => { const error = new Error('denied'); error.name='NotAllowedError'; throw error; };
  await sandbox.startCamera();
  assert(messages.some(x => x.error && x.text.includes('Permite la cámara')));
  devices = [];
  await sandbox.refreshCameras(); assert($('camera-device-status').textContent.includes('No hay cámaras'));
  devices = [{kind:'videoinput', deviceId:'usb', label:'USB'}];
  sandbox.navigator.mediaDevices.getUserMedia = async c => { constraints=c; return streamMock(); };
  await sandbox.startCamera();
  assert.equal(constraints.video.facingMode, 'user');
  devices = []; await sandbox.refreshCameras();
  assert.equal(stopped, 2);
  assert(messages.some(x => x.error && x.text.includes('desconectó')));
  // Late permission grant after cancellation must release the acquired camera.
  let grant;
  sandbox.navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { grant = resolve; });
  const pending = sandbox.startCamera(); await sandbox.stopCamera(); grant(streamMock()); await pending;
  assert.equal(stopped, 3);
  console.log('Cámaras: enumeración, selección exacta, desconexión, permiso denegado y cancelación comprobados.');
})().catch(error => { console.error(error); process.exitCode=1; });
