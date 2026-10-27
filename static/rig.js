'use strict';
// Las coordenadas ya vienen reflejadas del servidor, igual que la vista en espejo.
class RigOverlay {
  constructor(canvas) { this.canvas = canvas; this.enabled = false; this.timeout = null; }
  clear() {
    clearTimeout(this.timeout);
    this.canvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.canvas.hidden = true;
  }
  setEnabled(enabled) { this.enabled = !!enabled; this.clear(); }
  draw(rig) {
    this.clear();
    if (!this.enabled || !rig || !(rig.width > 0 && rig.height > 0)) return;
    const canvas = this.canvas;
    canvas.width = rig.width; canvas.height = rig.height; canvas.hidden = false;
    const context = canvas.getContext('2d');
    const hands = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],
      [9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
    const body = [[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
      [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24]];
    const valid = point => point && point.length === 2 && point.every(Number.isFinite);
    const drawGroup = (points = [], edges, color, radius) => {
      context.strokeStyle = color; context.fillStyle = color; context.lineWidth = 2;
      for (const [a, b] of edges) {
        if (!valid(points[a]) || !valid(points[b])) continue;
        context.beginPath(); context.moveTo(points[a][0]*canvas.width, points[a][1]*canvas.height);
        context.lineTo(points[b][0]*canvas.width, points[b][1]*canvas.height); context.stroke();
      }
      for (const point of points) {
        if (!valid(point)) continue;
        context.beginPath(); context.arc(point[0]*canvas.width, point[1]*canvas.height, radius, 0, Math.PI*2); context.fill();
      }
    };
    drawGroup(rig.pose, body, '#c6f16c', 3);
    drawGroup(rig.face, [], '#e8f7e0', 1.5);
    drawGroup(rig.left, hands, '#54dded', 3);
    drawGroup(rig.right, hands, '#ffbd69', 3);
    this.timeout = setTimeout(() => this.clear(), 600); // No dejar un esqueleto congelado.
  }
}
if (typeof module !== 'undefined') module.exports = RigOverlay;
