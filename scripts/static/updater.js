// ─────────────────────────────────────────────────────────────────────────────
// 3D plot setup
// ─────────────────────────────────────────────────────────────────────────────
const camera = {
    eye: { x: 2, y: 0, z: 0.5},
    center: { x: 0, y: 0, z: 0 },
    up: { x: 0, y: 0, z: 1 },
    projection: { type: 'perspective' }
};

const scene = {
    camera: camera,
    aspectmode: 'cube', // 'cube' makes axes equal (x=y=z). Use 'data' to match data ranges.
    xaxis: { title: 'X', backgroundcolor: '#fff', gridcolor: '#ddd', zerolinecolor: '#444', range: [-3, 3], nticks: 5 },
    yaxis: { title: 'Y', backgroundcolor: '#fff', gridcolor: '#ddd', zerolinecolor: '#444', range: [-3, 3], nticks: 5 },
    zaxis: { title: 'Z', backgroundcolor: '#fff', gridcolor: '#ddd', zerolinecolor: '#444', range: [-3, 3], nticks: 5 }
};

const size = 1000;
const layout = {
    width: size + 400,
    height: size,
    margin: { l: 20, r: 20, t: 20, b: 20 },
    scene: scene,
    scene_dragmode: 'orbit',
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff'
};

const socket = io();   // single connection for everything


// ─────────────────────────────────────────────────────────────────────────────
// 3D plot update
// ─────────────────────────────────────────────────────────────────────────────
let plotInitialized = false;
let rula_score = 0;

function capsuleMesh(p1, p2, radius, opacity, rula = true, segments = 16) {
    const x = [], y = [], z = [];
    const iIdx = [], jIdx = [], kIdx = [];

    // Axis
    const ax = p2[0] - p1[0], ay = p2[1] - p1[1], az = p2[2] - p1[2];
    const len = Math.sqrt(ax * ax + ay * ay + az * az);
    const ux = ax / len, uy = ay / len, uz = az / len;

    // Perpendicular basis
    let vx, vy, vz;
    if (Math.abs(ux) < 0.9) { vx = 0; vy = -uz; vz = uy; }
    else                     { vx = -uz; vy = 0; vz = ux; }
    const vl = Math.sqrt(vx * vx + vy * vy + vz * vz);
    vx /= vl; vy /= vl; vz /= vl;
    const wx = uy * vz - uz * vy, wy = uz * vx - ux * vz, wz = ux * vy - uy * vx;

    const rings = [];
    const capSegs = 8; // latitude steps per hemisphere

    // --- Back hemisphere (at p1, pointing away from p2) ---
    for (let si = capSegs; si >= 0; si--) {
        const phi = (Math.PI / 2) * (si / capSegs); // π/2 → 0
        const r = radius * Math.cos(phi);
        const h = -radius * Math.sin(phi);           // negative = behind p1
        const ring = [];
        for (let ti = 0; ti < segments; ti++) {
            const theta = (2 * Math.PI * ti) / segments;
            ring.push([
                p1[0] + h * ux + r * (Math.cos(theta) * vx + Math.sin(theta) * wx),
                p1[1] + h * uy + r * (Math.cos(theta) * vy + Math.sin(theta) * wy),
                p1[2] + h * uz + r * (Math.cos(theta) * vz + Math.sin(theta) * wz)
            ]);
        }
        rings.push(ring);
    }

    // --- Front hemisphere (at p2, pointing away from p1) ---
    for (let si = 0; si <= capSegs; si++) {
        const phi = (Math.PI / 2) * (si / capSegs); // 0 → π/2
        const r = radius * Math.cos(phi);
        const h = radius * Math.sin(phi);            // positive = beyond p2
        const ring = [];
        for (let ti = 0; ti < segments; ti++) {
            const theta = (2 * Math.PI * ti) / segments;
            ring.push([
                p2[0] + h * ux + r * (Math.cos(theta) * vx + Math.sin(theta) * wx),
                p2[1] + h * uy + r * (Math.cos(theta) * vy + Math.sin(theta) * wy),
                p2[2] + h * uz + r * (Math.cos(theta) * vz + Math.sin(theta) * wz)
            ]);
        }
        rings.push(ring);
    }

    // Flatten vertices
    for (const ring of rings) {
        for (const [px, py, pz] of ring) {
            x.push(px); y.push(py); z.push(pz);
        }
    }

    // Triangulate between adjacent rings
    const nRings = rings.length;
    for (let ri = 0; ri < nRings - 1; ri++) {
        for (let ti = 0; ti < segments; ti++) {
            const a = ri * segments + ti;
            const b = ri * segments + (ti + 1) % segments;
            const c = (ri + 1) * segments + ti;
            const d = (ri + 1) * segments + (ti + 1) % segments;
            iIdx.push(a); jIdx.push(b); kIdx.push(c);
            iIdx.push(b); jIdx.push(d); kIdx.push(c);
        }
    }

    let caps_color = '#aaaaaa';
    if (rula) {
        switch (rula_score) {
            case 1:
                caps_color = '#9CCB3B';
                break;
            case 2:
                caps_color = '#9CCB3B';
                break;
            case 3:
                caps_color = '#FAF04C';
                break;
            case 4:
                caps_color = '#FAF04C';
                break;
            case 5:
                caps_color = '#F3714D';
                break;
            case 6:
                caps_color = '#F3714D';
                break;
            case 7:
                caps_color = '#F04C4E';
                break;
            default:
                caps_color = '#aaaaaa';
        }
    }

    return {
        type: 'mesh3d',
        x, y, z,
        i: iIdx, j: jIdx, k: kIdx,
        opacity: opacity,
        color: caps_color,
        flatshading: false,
        lighting: { diffuse: 0.8, specular: 0.2 }
    };
}

function update_plot() {
    socket.on('update_plot', function (point) {
        const data = [];
        const COCO_SKELETON = [[0, 9], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [9, 10], [11, 13], [12, 14], [13, 15], [14, 16], [17, 19], [18, 20]];
        const r_sw_h = [0.16, 0.06, 0.06, 0.06, 0.06, 0.1, 0.1, 0.15, 0.1, 0.1, 0.08, 0.08, 0.05, 0.05];
        const ROBOT_CONFIG = [[0, 1], [2, 3], [4, 5], [6, 7]];
        const r_sw_r = point.radius;
        rv = [0.085, 0.085, 0.06, 0.065];
        
        for (let i = 0; i < COCO_SKELETON.length; i++) {
            const a = COCO_SKELETON[i][0];
            const b = COCO_SKELETON[i][1];
            if (point.x[a] != null && point.x[b] != null) {
                data.push(capsuleMesh([point.x[a], point.y[a], point.z[a]], [point.x[b], point.y[b], point.z[b]], r_sw_h[i], 0.4));
            }
        }

        if (point.x_robot && point.y_robot && point.z_robot && point.x_robot) {
            for (let i = 0; i < ROBOT_CONFIG.length; i++) {
                const a = ROBOT_CONFIG[i][0];
                const b = ROBOT_CONFIG[i][1];
                if (point.x_robot[a] != null && point.x_robot[b] != null) {
                    data.push(capsuleMesh([point.x_robot[a], point.y_robot[a], point.z_robot[a]], [point.x_robot[b], point.y_robot[b], point.z_robot[b]], r_sw_r[i], 0.4, rule = false));
                    data.push(capsuleMesh([point.x_robot[a], point.y_robot[a], point.z_robot[a]], [point.x_robot[b], point.y_robot[b], point.z_robot[b]], rv[i], 0.2, rule = false));
                }
            }
        }

        if (!plotInitialized) {
            Plotly.newPlot('plot', data, layout);
            plotInitialized = true;
        } else {
            Plotly.react('plot', data, layout);
        }
    });

    
}


// function update_robot() {
//     socket.on('update_robot', function (point) {
//         const ROBOT_CONFIG = [[0, 1], [2, 3], [4, 5], [6, 7]];
//         const r_sw_r = point.radius;
//         const data = [];
//         for (let i = 0; i < ROBOT_CONFIG.length; i++) {
//             const a = ROBOT_CONFIG[i][0];
//             const b = ROBOT_CONFIG[i][1];
//             if (point.x[a] != null && point.x[b] != null) {
//                 data.push(capsuleMesh([point.x[a], point.y[a], point.z[a]], [point.x[b], point.y[b], point.z[b]], r_sw_r[i]));
//             }
//         }
// 
//         if (!plotInitialized) {
//             Plotly.newPlot('plot', data, layout);
//             plotInitialized = true;
//         } else {
//             Plotly.react('plot', data, layout);
//         }
//     });
// }


// ─────────────────────────────────────────────────────────────────────────────
// RULA score update
// ─────────────────────────────────────────────────────────────────────────────
function update_rula() {
    socket.on('update_rula', function (score) {
        rula_score = Math.max(...score);
        const rulaDiv = document.getElementById('rula_score');
        rulaDiv.innerHTML = `RULA Score: ${rula_score}`;
    });
}


// ─────────────────────────────────────────────────────────────────────────────
// Streaming updates
// ─────────────────────────────────────────────────────────────────────────────
function update_stream1() {
    socket.on('update_stream1', function (data) {
        document.getElementById('image1').src = data.frame;
    });
}

function update_stream2() {
    socket.on('update_stream2', function (data) {
        document.getElementById('image2').src = data.frame;
    });
}

function update_stream3() {
    socket.on('update_stream3', function (data) {
        document.getElementById('image3').src = data.frame;
    });
}

function update_stream4() {
    socket.on('update_stream4', function (data) {
        document.getElementById('image4').src = data.frame;
    });
}


// ─────────────────────────────────────────────────────────────────────────────
// Launch functions @ 30Hz update rate
// ─────────────────────────────────────────────────────────────────────────────
setTimeout(update_plot, 1 / 30 * 1000);
setTimeout(update_rula, 1 / 30 * 1000);
setTimeout(update_stream1, 1 / 30 * 1000);
setTimeout(update_stream2, 1 / 30 * 1000);
setTimeout(update_stream3, 1 / 30 * 1000);
setTimeout(update_stream4, 1 / 30 * 1000);