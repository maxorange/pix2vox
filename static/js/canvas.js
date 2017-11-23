var Canvas = function (recorder) {
  this.recorder = recorder;
  this.exporter = new THREE.STLBinaryExporter();
  this.points = [];
  this.strokeStyle = '#48c9b0';
  this.lineWidth = 1;
  this.drawMode = 'outline';
  this.mousedown = false;
  this.container = document.getElementById('canvas');
  this.domElement = document.createElement('canvas');
  this.width = this.domElement.width;
  this.scale = 1. / this.width;
  this.container.appendChild(this.domElement);
};

Canvas.prototype.resize = function () {
  this.domElement.width = this.container.clientWidth;
  this.domElement.height = this.container.clientHeight;
  this.width = this.container.clientWidth;
  this.scale = 1. / this.width;
};

Canvas.prototype.draw = function () {
  var ctx = this.ctx = this.domElement.getContext('2d');
  ctx.lineCap = 'round';
  ctx.fillStyle = '#202b31';
  ctx.fillRect(0, 0, this.domElement.width, this.domElement.height);
  this.recorder.draw(ctx, this.width);
  ctx.lineWidth = this.lineWidth;
  ctx.strokeStyle = this.strokeStyle;
};

Canvas.prototype.line = function (startX, startY, endX, endY) {
  var ctx = this.ctx;
  drawLine(ctx, startX, startY, endX, endY);
  this.points.push([endX*this.scale, endY*this.scale]);
};

Canvas.prototype.record = function () {
  var stroke = this.points;
  var color = this.strokeStyle;
  var width = this.lineWidth*this.scale;
  var mode = this.drawMode;
  this.recorder.save(stroke, color, width, mode);
  this.points = [];
};

Canvas.prototype.clear = function () {
  var ctx = this.ctx;
  var width = this.domElement.width;
  var height = this.domElement.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillRect(0, 0, width, height);
  this.recorder.reset();
};

Canvas.prototype.send = function () {
  $.ajax({
    method: 'POST',
    url: 'image',
    dataType: 'json',
    data: {
      image: JSON.stringify(this.recorder.getData()),
      label: JSON.stringify(categoryWeights),
    }
  }).done(function (res) {
    var vertices = [];
    var faces = [];
    var lenVertices = res.verts.length;
    var lenFaces = res.faces.length;

    for (var i = 0; i < lenVertices; i++) {
      var vert = res.verts[i];
      vertices.push(new THREE.Vector3(vert[0], vert[1], vert[2]));
    }

    for (var i = 0; i < lenFaces; i++) {
      var face = res.faces[i];
      faces.push(new THREE.Face3(face[0], face[1], face[2]));
    }

    mesh.geometry.vertices = vertices;
    mesh.geometry.faces = faces;
    mesh.geometry.verticesNeedUpdate = true;
    mesh.geometry.elementsNeedUpdate = true;
    mesh.geometry.computeBoundingSphere();
    mesh.geometry.computeFaceNormals();
    mesh.geometry.center();

    var scale = 0.8 / mesh.geometry.boundingSphere.radius;
    mesh.scale.set(scale, scale, scale);
  });
};

Canvas.prototype.export = function (filename) {
  this.domElement.toBlob(function(blob){
    saveAs(blob, filename + '.png');
  });
};

Canvas.prototype.setDrawMode = function (mode) {
  this.drawMode = mode;
  var $slider = $('.slider.brightness');
  if (mode == 'depthmap') {
    $slider.slider('enable');
    var v = $slider.slider('value');
    this.strokeStyle = rgbToHex(v, v, v);
    this.lineWidth = 10;
  } else {
    $slider.slider('disable');
    this.strokeStyle = '#48c9b0';
    this.lineWidth = 1;
  }
  $('[name=lineWidth]').val(this.lineWidth);
  this.draw();
};

Canvas.prototype.setLineWidth = function (width) {
  this.lineWidth = width;
  this.draw();
};

Canvas.prototype.setStrokeStyle = function (hex) {
  this.strokeStyle = hex;
  this.draw();
};
