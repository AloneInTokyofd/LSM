'use strict';
// Control de una tanda: un resultado repetido nunca cuenta como otra toma.
class ContinuousCapture {
  constructor() { this.active = false; this.count = 0; this.limit = 0; this.serial = 0; this.ids = new Set(); }
  start(limit) {
    if (!Number.isInteger(limit) || limit < 5 || limit > 60) throw new Error('Elige entre 5 y 60 tomas.');
    if (this.active) throw new Error('Ya hay una tanda en curso.');
    this.active = true; this.count = 0; this.limit = limit; this.ids.clear();
    return ++this.serial;
  }
  complete(id, serial) {
    if (!this.active || serial !== this.serial || this.ids.has(id)) return null;
    this.ids.add(id); this.count++;
    const done = this.count >= this.limit;
    if (done) this.active = false;
    return {count: this.count, limit: this.limit, done};
  }
  stop() { this.active = false; this.serial++; }
  current(serial) { return this.active && serial === this.serial; }
}
if (typeof module !== 'undefined') module.exports = ContinuousCapture;
