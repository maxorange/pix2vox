var canvas;

$(window).on('load', function() {
  var recorder = new Recorder();
  canvas = new Canvas(recorder);

  if (!canvas.domElement || !canvas.domElement.getContext) return false;

  canvas.resize();
  canvas.draw();

  var startX, startY;
  var endX, endY;

  $(canvas.domElement).on('mousedown', function(e) {
    startX = e.pageX - $(this).offset().left;
    startY = e.pageY - $(this).offset().top;
    canvas.points.push([startX*canvas.scale, startY*canvas.scale]);
    canvas.mousedown = true;
  });

  $(canvas.domElement).on('mousemove', function(e) {
    if (!canvas.mousedown) return;
    endX = e.pageX - $(this).offset().left;
    endY = e.pageY - $(this).offset().top;
    canvas.line(startX, startY, endX, endY);
    startX = endX;
    startY = endY;
  });

  $(canvas.domElement).on('mouseup', function(e) {
    canvas.mousedown = false;
    canvas.record();
    canvas.draw();
    canvas.send();
  });

  $(canvas.domElement).on('mouseleave', function(e) {
    if (!canvas.mousedown) return;
    canvas.mousedown = false;
    canvas.record();
    canvas.draw();
  });

  $(canvas.domElement).on('wheel', function(e) {
    var width = canvas.lineWidth;
    width += Math.sign(e.originalEvent.deltaY);
    width = Math.max(1, Math.min(width, 20));
    canvas.setLineWidth(width);
    $('[name=lineWidth]').val(width);
  });

  $('[name=lineWidth]').on('change', function(e) {
    canvas.setLineWidth(e.target.value);
  });

  $('[name=export]').on('click', function(e) {
    var filename = $('[name=fileName]').val();
    canvas.export(filename);
    exportSTLBinary(filename);
  });

  $('[name=clear]').on('click', function(e) {
    canvas.clear();
  });

  $('[name=drawMode]').on('change', function(e) {
    var mode = $(e.target).val();
    canvas.setDrawMode(mode);
  });

  $('.slider.category-weights').slider({
    max: 100,
    min: 0,
    step: 1,
    value: 0,
    orientation: 'horizontal',
    range: 'min',
    change: function(e, ui) {
      var id = $(e.target).data('id');
      var value = ui.value;
      categoryWeights[id] = value*0.01;
      canvas.send();
    }
  });

  $('.slider.brightness').slider({
    max: 255,
    min: 1,
    step: 1,
    value: 255,
    disabled: true,
    orientation: 'horizontal',
    slide: function(e, ui) {
      var v = ui.value;
      var hex = rgbToHex(v, v, v);
      canvas.setStrokeStyle(hex);
      $(e.target).css('background-color', hex);
    }
  });

  $('.input-group').on('focus', '.form-control', function () {
    $(this).closest('.input-group, .form-group').addClass('focus');
  }).on('blur', '.form-control', function () {
    $(this).closest('.input-group, .form-group').removeClass('focus');
  });

  $('[data-toggle="checkbox"]').radiocheck();
  $('[data-toggle="radio"]').radiocheck();
  $('[data-toggle=tooltip]').tooltip();
});

$(window).on('resize', function() {
  canvas.resize();
  canvas.draw();
});
