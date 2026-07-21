/* =========================================================
   Panda modified-DH parameters (Franka Emika official table)
   columns: a_{i-1} [m], alpha_{i-1} [rad], d_i [m]
   Verified against franka_ros/franka_description xacro joint origins.
   ========================================================= */
const DH = [
  { a:0,       alpha:0,          d:0.333, name:'J1', mesh:'link1' },
  { a:0,       alpha:-Math.PI/2, d:0,     name:'J2', mesh:'link2' },
  { a:0,       alpha: Math.PI/2, d:0.316, name:'J3', mesh:'link3' },
  { a:0.0825,  alpha: Math.PI/2, d:0,     name:'J4', mesh:'link4' },
  { a:-0.0825, alpha:-Math.PI/2, d:0.384, name:'J5', mesh:'link5' },
  { a:0,       alpha: Math.PI/2, d:0,     name:'J6', mesh:'link6' },
  { a:0.088,   alpha: Math.PI/2, d:0,     name:'J7', mesh:'link7' },
  { a:0,       alpha:0,          d:0.107, name:'Flange', fixed:true, mesh:null }
];

const LIMITS = [
  [-2.8973, 2.8973],
  [-1.7628, 1.7628],
  [-2.8973, 2.8973],
  [-3.0718, -0.0698],
  [-2.8973, 2.8973],
  [-0.0175, 3.7525],
  [-2.8973, 2.8973]
];

const READY_POSE = [0, -Math.PI/4, 0, -3*Math.PI/4, 0, Math.PI/2, Math.PI/4];
const ZERO_POSE  = [0, 0, 0, -Math.PI/2, 0, Math.PI/2, 0];

let q = READY_POSE.slice();
let gripperOpening = 0.04;
let HAND = false;

/* warn early if opened directly as a file:// page — ColladaLoader needs XHR,
   which browsers block on the file:// protocol. */
if (location.protocol === 'file:') {
  document.getElementById('loadingTxt').innerHTML =
    'Open this over a local server, not file://<br>e.g. <code>python3 -m http.server</code> in this folder';
  document.querySelector('#loading .spin').style.display = 'none';
}

/* ---------- three.js scaffold ---------- */
const app = document.getElementById('app');
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

const pedestalMat = new THREE.MeshStandardMaterial({ color:0x1c1f1c, metalness:0.4, roughness:0.5 });
const robotGroup = new THREE.Group();
scene.add(robotGroup);
const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.1,0.12,0.04,32), pedestalMat);
pedestal.position.y = 0.02;
robotGroup.add(pedestal);

/* ---------- load real Franka visual meshes (.dae) ---------- */
var MESH_FILES = [];
if (HAND) {
  MESH_FILES = ['link0','link1','link2','link3','link4','link5','link6','link7','hand','finger'];
}
else {
  MESH_FILES = ['link0','link1','link2','link3','link4','link5','link6','link7'];
}

const loader = new THREE.ColladaLoader();
const loaded = {};

function loadOne(name){
  return new Promise((resolve, reject)=>{
    loader.load(
      `meshes/visual/${name}.dae`,
      (collada) => {
        const wrapper = new THREE.Group();
        collada.scene.quaternion.identity();
        collada.scene.position.set(0,0,0);
        collada.scene.scale.set(1,1,1);
        collada.scene.traverse(o=>{
          if (o.isMesh){
            o.castShadow = false;
            o.receiveShadow = false;
          }
        });
        wrapper.add(collada.scene);
        loaded[name] = wrapper;
        robotGroup.add(wrapper);
        resolve();
      },
      undefined,
      (err) => reject(err)
    );
  });
}

const loadingTxt = document.getElementById('loadingTxt');
let doneCount = 0;
Promise.all(MESH_FILES.map(n =>
  loadOne(n).then(()=>{
    doneCount++;
    loadingTxt.textContent = `Loading Franka mesh library… (${doneCount}/${MESH_FILES.length})`;
  })
)).then(()=>{
  // build the two gripper fingers from the single finger.dae (as the real robot does)
  if (HAND) {
    const fingerTemplate = loaded['finger'];
    robotGroup.remove(fingerTemplate);
    const fingerL = fingerTemplate;
    const fingerR = fingerTemplate.clone(true);
    robotGroup.add(fingerL, fingerR);
    loaded['fingerL'] = fingerL;
    loaded['fingerR'] = fingerR;
  }

  document.getElementById('loading').style.opacity = '0';
  setTimeout(()=> document.getElementById('loading').style.display = 'none', 300);

  buildJointUI();
  updateKinematics();
  animate();
}).catch(err=>{
  loadingTxt.innerHTML = 'Failed to load meshes.<br>Check the browser console and that meshes/visual/*.dae exist.';
  document.querySelector('#loading .spin').style.display = 'none';
  console.error(err);
});

const frameHelpers = [];
for (let i=0;i<DH.length;i++){
  const f = new THREE.AxesHelper(0.09);
  f.visible = false;
  robotGroup.add(f);
  frameHelpers.push(f);
}
const flangeHelper = new THREE.AxesHelper(0.09);
flangeHelper.visible = false;
robotGroup.add(flangeHelper);

/* ---------- forward kinematics ---------- */
function dhMatrix(a, alpha, d, theta){
  const ca=Math.cos(alpha), sa=Math.sin(alpha);
  const ct=Math.cos(theta), st=Math.sin(theta);
  // modified DH (Craig convention): Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)
  const m = new THREE.Matrix4();
  m.set(
    ct,      -st,      0,     a,
    st*ca,   ct*ca,   -sa,   -sa*d,
    st*sa,   ct*sa,    ca,    ca*d,
    0,        0,        0,     1
  );
  return m;
}

function updateKinematics(){
  const baseRot = new THREE.Matrix4().makeRotationX(-Math.PI/2);
  let T = baseRot.clone();
  T.setPosition(0, 0.04, 0);

  if (loaded['link0']){
    loaded['link0'].position.setFromMatrixPosition(T);
    loaded['link0'].quaternion.setFromRotationMatrix(T);
  }

  const mats = [T.clone()];
  for (let i=0;i<DH.length;i++){
    const theta = DH[i].fixed ? 0 : q[i];
    T = T.clone().multiply(dhMatrix(DH[i].a, DH[i].alpha, DH[i].d, theta));
    mats.push(T.clone());
    if (DH[i].mesh && loaded[DH[i].mesh]){
      const obj = loaded[DH[i].mesh];
      obj.position.setFromMatrixPosition(T);
      obj.quaternion.setFromRotationMatrix(T);
    }
    const fh = frameHelpers[i];
    fh.position.setFromMatrixPosition(T);
    fh.quaternion.setFromRotationMatrix(T);
  }

  // hand attaches to the flange with a fixed -45deg twist (franka_hand.xacro)
  const flangeMat = mats[mats.length-1];
  const handMat4 = flangeMat.clone().multiply(new THREE.Matrix4().makeRotationZ(-Math.PI/4));
  if (loaded['hand']){
    loaded['hand'].position.setFromMatrixPosition(handMat4);
    loaded['hand'].quaternion.setFromRotationMatrix(handMat4);
  }
  flangeHelper.position.setFromMatrixPosition(handMat4);
  flangeHelper.quaternion.setFromRotationMatrix(handMat4);

  // fingers: prismatic joints at hand-frame z=0.0584, sliding along hand-local Y.
  // the right finger's visual mesh is authored pre-rotated 180deg about Z (see franka_hand.xacro)
  // so the same finger.dae can be reused mirrored for both sides.
  const fingerBase = new THREE.Matrix4().makeTranslation(0, 0, 0.0584);
  const leftMat = handMat4.clone().multiply(fingerBase).multiply(new THREE.Matrix4().makeTranslation(0, gripperOpening, 0));
  const rightMat = handMat4.clone().multiply(fingerBase)
    .multiply(new THREE.Matrix4().makeTranslation(0, -gripperOpening, 0))
    .multiply(new THREE.Matrix4().makeRotationZ(Math.PI));
  if (loaded['fingerL']){
    loaded['fingerL'].position.setFromMatrixPosition(leftMat);
    loaded['fingerL'].quaternion.setFromRotationMatrix(leftMat);
  }
  if (loaded['fingerR']){
    loaded['fingerR'].position.setFromMatrixPosition(rightMat);
    loaded['fingerR'].quaternion.setFromRotationMatrix(rightMat);
  }

  // TCP readout — hand frame, 0.1034m along its z (franka_hand default tcp offset)
  const tcpMat = handMat4.clone().multiply(new THREE.Matrix4().makeTranslation(0,0,0.1034));
  const p = new THREE.Vector3().setFromMatrixPosition(tcpMat);
  document.getElementById('readout').innerHTML =
    `x <b>${p.x.toFixed(3)}</b> m<br>y <b>${p.z.toFixed(3)}</b> m<br>z (up) <b>${p.y.toFixed(3)}</b> m`;
}

/* ---------- UI: joint sliders (built after meshes finish loading) ---------- */
const sliders = [];
function buildJointUI(){
  const jointsDiv = document.getElementById('joints');
  LIMITS.forEach((lim,i)=>{
    const row = document.createElement('div');
    row.className = 'joint-row';
    row.innerHTML = `
      <div class="label-line"><span class="jname">${DH[i].name}</span><span class="jval" id="jval${i}">0.0°</span></div>
      <input type="range" id="jslider${i}" min="${lim[0]}" max="${lim[1]}" step="0.005" value="${q[i]}">
    `;
    jointsDiv.appendChild(row);
    const slider = row.querySelector('input');
    slider.addEventListener('input', ()=>{
      q[i] = parseFloat(slider.value);
      document.getElementById(`jval${i}`).textContent = (q[i]*180/Math.PI).toFixed(1)+'°';
      updateKinematics();
    });
    sliders.push(slider);
  });
}

function applyPose(pose){
  q = pose.slice();
  q.forEach((v,i)=>{
    sliders[i].value = v;
    document.getElementById(`jval${i}`).textContent = (v*180/Math.PI).toFixed(1)+'°';
  });
  updateKinematics();
}

document.getElementById('btnReady').addEventListener('click', ()=>applyPose(READY_POSE));
document.getElementById('btnZero').addEventListener('click', ()=>applyPose(ZERO_POSE));
document.getElementById('btnRandom').addEventListener('click', ()=>{
  const rp = LIMITS.map(([lo,hi])=> lo + Math.random()*(hi-lo));
  applyPose(rp);
});

const gripSlider = document.getElementById('gripSlider');
const gripVal = document.getElementById('gripVal');
gripSlider.addEventListener('input', ()=>{
  gripperOpening = parseFloat(gripSlider.value);
  gripVal.textContent = (gripperOpening*2*1000).toFixed(0)+' mm';
  updateKinematics();
});

document.getElementById('toggleGrid').addEventListener('change', e=>{ grid.visible = e.target.checked; });
document.getElementById('toggleFrames').addEventListener('change', e=>{
  frameHelpers.forEach(f=>f.visible = e.target.checked);
  flangeHelper.visible = e.target.checked;
});
let spin = false;
document.getElementById('toggleSpin').addEventListener('change', e=>{ spin = e.target.checked; });

/* ---------- camera orbit (manual, no external deps) ---------- */
let camDist = 1.7, camTheta = 0.9, camPhi = 1.15, camTarget = new THREE.Vector3(0,0.35,0);
let dragging = false, panning = false, lastX=0, lastY=0;

function updateCamera(){
  const x = camTarget.x + camDist*Math.sin(camPhi)*Math.sin(camTheta);
  const y = camTarget.y + camDist*Math.cos(camPhi);
  const z = camTarget.z + camDist*Math.sin(camPhi)*Math.cos(camTheta);
  camera.position.set(x,y,z);
  camera.lookAt(camTarget);
}

renderer.domElement.addEventListener('mousedown', e=>{
  if (e.button === 2) panning = true; else dragging = true;
  lastX = e.clientX; lastY = e.clientY;
});
window.addEventListener('mouseup', ()=>{ dragging=false; panning=false; });
window.addEventListener('mousemove', e=>{
  const dx = e.clientX-lastX, dy = e.clientY-lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (dragging){
    camTheta -= dx*0.006;
    camPhi = Math.min(Math.max(camPhi - dy*0.006, 0.15), Math.PI-0.1);
    updateCamera();
  } else if (panning){
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix,0);
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix,1);
    camTarget.addScaledVector(right, -dx*0.0015*camDist);
    camTarget.addScaledVector(up, dy*0.0015*camDist);
    updateCamera();
  }
});
renderer.domElement.addEventListener('contextmenu', e=>e.preventDefault());
renderer.domElement.addEventListener('wheel', e=>{
  e.preventDefault();
  camDist = Math.min(Math.max(camDist * (1 + e.deltaY*0.001), 0.5), 5);
  updateCamera();
}, { passive:false });

/* ---------- render loop ---------- */
function animate(){
  requestAnimationFrame(animate);
  if (spin){ camTheta += 0.003; updateCamera(); }
  renderer.render(scene, camera);
}

resize();
updateCamera();
