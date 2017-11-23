if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

var container;
var camera, cameraTarget, scene, renderer, mesh;
var queries = parseQuery();
var rad90 = 90*THREE.Math.DEG2RAD;

window.onload = function() {
  init();
  animate();
};

function init() {
  container = document.getElementById('viewer');

  camera = new THREE.PerspectiveCamera( 35, container.clientWidth / container.clientHeight, 0.1, 15 );
  camera.position.set(2, 0.15, 2);
  cameraTarget = new THREE.Vector3(0, 0, 0);

  controls = new THREE.TrackballControls(camera, container);
	controls.rotateSpeed = 3.0;
	controls.zoomSpeed = 1.2;
	controls.panSpeed = 0.8;
	controls.noZoom = false;
	controls.noPan = false;
	controls.staticMoving = true;
	controls.dynamicDampingFactor = 0.3;
	controls.keys = [65, 83, 68];
	controls.addEventListener('change', render);

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x202b31, 2, 15);

  var geometry = new THREE.Geometry();
  var material = new THREE.MeshPhongMaterial({color:0xffffff});
  mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.set(0, 0, -rad90);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);

  // Lights
  scene.add(new THREE.HemisphereLight(0x443333, 0x111122));
  addShadowedLight(1, 1, 1, 0xdddddd, 0.5);
  addShadowedLight(0.5, 1, -1, 0xaaaaaa, 1);

  // renderer
  renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
  renderer.setClearColor(scene.fog.color);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.gammaInput = true;
  renderer.gammaOutput = true;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.renderReverseSided = false;
  container.appendChild(renderer.domElement);

  window.addEventListener('resize', onWindowResize, false);
  // document.getElementById('screenshot').addEventListener('click', screenshot, false);
}

function addShadowedLight( x, y, z, color, intensity ) {
  var directionalLight = new THREE.DirectionalLight( color, intensity );
  directionalLight.position.set( x, y, z );
  scene.add( directionalLight );

  directionalLight.castShadow = true;

  var d = 1;
  directionalLight.shadow.camera.left = -d;
  directionalLight.shadow.camera.right = d;
  directionalLight.shadow.camera.top = d;
  directionalLight.shadow.camera.bottom = -d;

  directionalLight.shadow.camera.near = 1;
  directionalLight.shadow.camera.far = 4;

  directionalLight.shadow.mapSize.width = 1024;
  directionalLight.shadow.mapSize.height = 1024;

  directionalLight.shadow.bias = -0.005;
}

function onWindowResize() {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
  controls.handleResize();
}

function animate() {
  requestAnimationFrame( animate );
  controls.update();
  render();
}

function render() {
  renderer.render( scene, camera );
}

function parseQuery(text, sep, eq, isDecode) {
  text = text || location.search.substr(1);
  sep = sep || '&';
  eq = eq || '=';
  var decode = (isDecode) ? decodeURIComponent : function(a) { return a; };
  return text.split(sep).reduce(function(obj, v) {
    var pair = v.split(eq);
    obj[pair[0]] = decode(pair[1]);
    return obj;
  }, {});
}

function screenshot() {
  renderer.domElement.toBlob(function(blob){
    saveAs(blob, queries.name + '.png');
  });
}

function exportSTLBinary(filename) {
  var data = new THREE.STLBinaryExporter().parse(scene);
  var blob = new Blob([data], {type: 'application/octet-binary'});
  saveAs(blob, filename + '.stl');
}
