/** Simple loss chart on a 2d canvas */

export class LossChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.losses = [];
    this.maxPoints = 300;
  }

  push(loss) {
    this.losses.push(loss);
    if (this.losses.length > this.maxPoints) this.losses.shift();
    this.draw();
  }

  reset() {
    this.losses = [];
    this.draw();
  }

  draw() {
    const c = this.canvas;
    const ctx = this.ctx;
    const w = c.width;
    const h = c.height;
    ctx.fillStyle = '#0a0e16';
    ctx.fillRect(0, 0, w, h);

    if (this.losses.length < 2) {
      ctx.fillStyle = '#8b9bb8';
      ctx.font = '11px sans-serif';
      ctx.fillText('Loss will appear when training…', 10, h / 2);
      return;
    }

    const min = Math.min(...this.losses);
    const max = Math.max(...this.losses);
    const span = max - min || 1e-6;

    ctx.strokeStyle = '#5b8cff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    this.losses.forEach((v, i) => {
      const x = (i / (this.losses.length - 1)) * (w - 8) + 4;
      const y = h - 6 - ((v - min) / span) * (h - 14);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = '#8b9bb8';
    ctx.font = '10px monospace';
    ctx.fillText(max.toFixed(3), 4, 12);
    ctx.fillText(min.toFixed(3), 4, h - 4);
  }
}
