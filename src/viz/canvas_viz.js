/**
 * Canvas renderer: data points, generated samples, trajectories, vector field.
 */

const WORLD = 2.2; // half-extent in data coords

export class CanvasViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.resize();
    this.showData = true;
    this.showGen = true;
    this.showField = false;
    this.showTraj = false;
    this.drawMode = false;
    this.drawnPoints = [];
    this._bindDraw();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const size = Math.min(rect.width || 640, 720);
    this.canvas.width = size * this.dpr;
    this.canvas.height = size * this.dpr;
    this.canvas.style.width = size + 'px';
    this.canvas.style.height = size + 'px';
    this.size = size;
  }

  worldToScreen(x, y) {
    const s = this.canvas.width;
    const sx = ((x + WORLD) / (2 * WORLD)) * s;
    const sy = ((WORLD - y) / (2 * WORLD)) * s;
    return [sx, sy];
  }

  screenToWorld(sx, sy) {
    const s = this.canvas.width;
    const x = (sx / s) * 2 * WORLD - WORLD;
    const y = WORLD - (sy / s) * 2 * WORLD;
    return [x, y];
  }

  clear() {
    const ctx = this.ctx;
    const s = this.canvas.width;
    ctx.fillStyle = '#0a0e16';
    ctx.fillRect(0, 0, s, s);
    // subtle grid
    ctx.strokeStyle = 'rgba(42,53,72,0.55)';
    ctx.lineWidth = 1 * this.dpr;
    const step = s / 8;
    for (let i = 1; i < 8; i++) {
      ctx.beginPath();
      ctx.moveTo(i * step, 0);
      ctx.lineTo(i * step, s);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * step);
      ctx.lineTo(s, i * step);
      ctx.stroke();
    }
    // axes
    ctx.strokeStyle = 'rgba(91,140,255,0.2)';
    ctx.beginPath();
    ctx.moveTo(s / 2, 0);
    ctx.lineTo(s / 2, s);
    ctx.moveTo(0, s / 2);
    ctx.lineTo(s, s / 2);
    ctx.stroke();
  }

  drawPoints(flatXY, color, radius = 2.2, alpha = 0.85) {
    if (!flatXY || flatXY.length < 2) return;
    const ctx = this.ctx;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    const r = radius * this.dpr;
    for (let i = 0; i < flatXY.length; i += 2) {
      const [sx, sy] = this.worldToScreen(flatXY[i], flatXY[i + 1]);
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  /** field: array of {x,y,vx,vy} or parallel arrays */
  drawVectorField(field, color = '#3dd68c') {
    if (!field || !field.length) return;
    const ctx = this.ctx;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.2 * this.dpr;
    const scale = 0.18 * this.canvas.width / (2 * WORLD);

    for (const p of field) {
      const [sx, sy] = this.worldToScreen(p.x, p.y);
      const len = Math.hypot(p.vx, p.vy) + 1e-6;
      const maxL = 0.35;
      const mag = Math.min(len, maxL) / maxL;
      const dx = (p.vx / len) * mag * 18 * this.dpr;
      const dy = -(p.vy / len) * mag * 18 * this.dpr; // screen y flip
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(sx + dx, sy + dy);
      ctx.stroke();
      // arrow head
      const ang = Math.atan2(dy, dx);
      ctx.beginPath();
      ctx.moveTo(sx + dx, sy + dy);
      ctx.lineTo(
        sx + dx - 5 * this.dpr * Math.cos(ang - 0.4),
        sy + dy - 5 * this.dpr * Math.sin(ang - 0.4)
      );
      ctx.lineTo(
        sx + dx - 5 * this.dpr * Math.cos(ang + 0.4),
        sy + dy - 5 * this.dpr * Math.sin(ang + 0.4)
      );
      ctx.closePath();
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  /** trajs: array of arrays of [x,y] */
  drawTrajectories(trajs, color = '#ffb020') {
    if (!trajs) return;
    const ctx = this.ctx;
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 1.1 * this.dpr;
    for (const path of trajs) {
      if (!path || path.length < 2) continue;
      ctx.beginPath();
      for (let i = 0; i < path.length; i++) {
        const [sx, sy] = this.worldToScreen(path[i][0], path[i][1]);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  render({ data, gen, field, trajs }) {
    this.clear();
    if (this.showTraj && trajs) this.drawTrajectories(trajs);
    if (this.showField && field) this.drawVectorField(field);
    if (this.showData && data) this.drawPoints(data, '#5b8cff', 2.4, 0.75);
    if (this.showGen && gen) this.drawPoints(gen, '#ff7ab6', 2.6, 0.9);
    if (this.drawMode && this.drawnPoints.length) {
      this.drawPoints(this.drawnPoints, '#7c5cff', 3.2, 0.95);
    }
  }

  _bindDraw() {
    const c = this.canvas;
    let drawing = false;

    const add = (e) => {
      if (!this.drawMode) return;
      const rect = c.getBoundingClientRect();
      const scaleX = c.width / rect.width;
      const scaleY = c.height / rect.height;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      const sx = (clientX - rect.left) * scaleX;
      const sy = (clientY - rect.top) * scaleY;
      const [x, y] = this.screenToWorld(sx, sy);
      // scatter a few points around stroke
      for (let k = 0; k < 3; k++) {
        this.drawnPoints.push(
          x + (Math.random() - 0.5) * 0.08,
          y + (Math.random() - 0.5) * 0.08
        );
      }
    };

    c.addEventListener('mousedown', (e) => {
      if (!this.drawMode) return;
      drawing = true;
      add(e);
    });
    c.addEventListener('mousemove', (e) => {
      if (drawing) add(e);
    });
    window.addEventListener('mouseup', () => { drawing = false; });
    c.addEventListener('touchstart', (e) => {
      if (!this.drawMode) return;
      drawing = true;
      add(e);
      e.preventDefault();
    }, { passive: false });
    c.addEventListener('touchmove', (e) => {
      if (drawing) { add(e); e.preventDefault(); }
    }, { passive: false });
    c.addEventListener('touchend', () => { drawing = false; });
  }

  clearDrawing() {
    this.drawnPoints = [];
  }

  getDrawnAsDataset(maxN = 512) {
    const pts = this.drawnPoints;
    if (pts.length < 4) return null;
    // subsample
    const n = Math.min(maxN, Math.floor(pts.length / 2));
    const out = new Array(n * 2);
    for (let i = 0; i < n; i++) {
      const j = ((i * Math.floor(pts.length / 2 / n)) % Math.floor(pts.length / 2)) * 2;
      out[i * 2] = pts[j];
      out[i * 2 + 1] = pts[j + 1];
    }
    return out;
  }
}
