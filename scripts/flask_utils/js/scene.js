/* warn early if opened directly as a file:// page — ColladaLoader needs XHR,
   which browsers block on the file:// protocol. */
if (location.protocol === 'file:') {
  document.getElementById('loadingTxt').innerHTML =
    'Open this over a local server, not file://<br>e.g. <code>python3 -m http.server</code> in this folder';
  document.querySelector('#loading .spin').style.display = 'none';
}

const socket = io();

/* ---------- three.js scaffold ---------- */
const app = document.getElementById('plot');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c0f0d);
scene.fog = new THREE.Fog(0x0c0f0d, 3, 9);

const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 50);
const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
app.appendChild(renderer.domElement);

function resize(){
  const w = app.clientWidth, h = app.clientHeight;
  renderer.setSize(w,h);
  camera.aspect = w/h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

/* lighting — a bit brighter than a PBR scene since Collada materials build as Phong */
scene.add(new THREE.HemisphereLight(0xaebdb4, 0x0a0a0a, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(1.6, 2.4, 1.2);
scene.add(key);
const fill = new THREE.DirectionalLight(0xbfd4ff, 0.4);
fill.position.set(-1.4, 1.0, 1.6);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xff8a3d, 0.25);
rim.position.set(-1.5, 1.0, -1.8);
scene.add(rim);

/* floor grid */
const grid = new THREE.GridHelper(3, 30, 0x2c3a32, 0x1a221d);
scene.add(grid);
const worldAxes = new THREE.AxesHelper(0.25);
worldAxes.position.set(0,0.002,0);
scene.add(worldAxes);

// const pedestalMat = new THREE.MeshStandardMaterial({ color:0x1c1f1c, metalness:0.4, roughness:0.5 });
// const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.1,0.12,0.04,32), pedestalMat);
// pedestal.position.y = 0.02;
// robotGroup.add(pedestal);
