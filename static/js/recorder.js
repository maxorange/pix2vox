var Recorder = function () {
  this.numElements = 0;
  this.strokes = [];
  this.colors = [];
  this.widths = [];
  this.modes = [];
};

Recorder.prototype.save = function (stroke, color, width, mode) {
  this.numElements += 1;
  this.strokes.push(stroke);
  this.colors.push(color);
  this.widths.push(width);
  this.modes.push(mode);
};

Recorder.prototype.reset = function () {
  this.numElements = 0;
  this.strokes = [];
  this.colors = [];
  this.widths = [];
  this.modes = [];
};

Recorder.prototype.draw = function (ctx, scale) {
  var numElements = this.numElements;
  var strokes = this.strokes;
  var colors = this.colors;
  var widths = this.widths;
  var modes = this.modes;

  for (var i = 0; i < numElements; i++) {
    ctx.lineWidth = widths[i]*scale;
    ctx.strokeStyle = colors[i];
    var points = strokes[i];
    var numPoints = points.length;
    for (var j = 0; j < numPoints-1; j++) {
      drawLine(ctx,
        points[j][0]*scale, points[j][1]*scale,
        points[j+1][0]*scale, points[j+1][1]*scale
      );
    }
  }
};

Recorder.prototype.getData = function () {
  return {
    strokes: this.strokes,
    colors: this.colors,
    widths: this.widths,
    modes: this.modes,
  }
};
