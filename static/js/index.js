// Comparison with baselines: the ground truth on the left, PINE in the middle and a baseline picked from a menu on
// the right, the task and the view above, the samples as thumbnails below. Point maps are shown as 3D point clouds
// (three.js) that share one camera, so the three views move together; the other tasks are images. The tiles, clouds,
// start views and per-sample errors come from compare_data.js, which teaser_render/export_web_compare.py writes.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA = window.COMPARE_DATA;
const ROOT = './static/images/compare/';
const ORDER = ['point', 'depth', 'normal', 'planarity', 'instance'];
const DEFAULT = { point: 'MoGe-3', depth: 'MoGe-3', normal: 'MoGe-3', planarity: 'MoGe-3', instance: 'gen2seg' };
const NOTE = {
  point: 'Drag to rotate, scroll to zoom, right-drag to pan; double-click to reset the view.',
  planarity: 'Every annotated plane is coloured by the signed distance to its fitted plane; white is flat.',
  instance: 'Masks for the prompt point, the red dot.',
};
const ERROR_NOTE = 'One colour scale per sample: blue is a low error and red a high one; black has no ground truth.';
const FOOT = 1.9;           // point size in pixels of the input image, for clouds thinned to 100k points
const BACKGROUND = 0xf4f4f2;

const state = { task: 'point', sample: 0, other: null, mode: 'pred' };
const $ = (id) => document.getElementById(id);
const label = (m) => DATA.labels[m] || m;
const task = () => DATA.tasks[state.task];
const sample = () => task().samples[state.sample];
const file = (s, method, ext = '.jpg') => ROOT + s.dir + '/' + method + ext;
const PANELS = { gt: 'GT', ours: DATA.ours, other: null };
const method = (key) => (key === 'other' ? state.other : PANELS[key]);

// ---------- 3D viewer: three canvases, one camera ----------

let viewer = null;

function makeViewer() {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000);
  const target = new THREE.Vector3();
  const material = new THREE.ShaderMaterial({
    uniforms: { uScale: { value: 1 }, uFoot: { value: 0.002 } },
    vertexShader: `
      uniform float uScale;
      uniform float uFoot;
      attribute vec3 color;
      varying vec3 vColor;
      void main() {
        vColor = color;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        // world size of a pixel footprint at the depth the point had in the input camera
        float size = max(-position.z, 0.0) * uFoot;
        gl_PointSize = clamp(size * uScale / -mv.z, 1.0, 24.0);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      void main() { gl_FragColor = vec4(vColor, 1.0); }`,
  });
  const views = {};
  let pending = false;
  const render = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      Object.values(views).forEach((v) => {
        material.uniforms.uScale.value = v.renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
        v.renderer.render(v.scene, camera);
      });
    });
  };
  Object.keys(PANELS).forEach((key) => {
    const box = document.querySelector('#panel-' + key + ' .compare-cloud');
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(BACKGROUND);
    box.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target = target;           // shared, so panning in one view pans all three
    controls.addEventListener('change', render);
    renderer.domElement.addEventListener('dblclick', () => reset());
    const scene = new THREE.Scene();
    views[key] = { box, renderer, controls, scene };
    new ResizeObserver(() => {
      const w = box.clientWidth;
      if (w > 0) { renderer.setSize(w, w, false); render(); }
    }).observe(box);
  });

  const cache = new Map();
  function load(url) {
    if (!cache.has(url)) {
      cache.set(url, fetch(url).then((r) => {
        if (!r.ok) throw new Error(url + ' ' + r.status);
        return r.arrayBuffer();
      }).then((buf) => {
        const n = new Uint32Array(buf, 0, 1)[0];
        const box = new Float32Array(buf, 4, 6);
        const q = new Uint16Array(buf, 28, 3 * n);
        const rgb = new Uint8Array(buf, 28 + 6 * n, 3 * n);
        const pos = new Float32Array(3 * n);
        for (let i = 0; i < 3 * n; i++) {
          const a = i % 3;
          pos[i] = box[a] + (q[i] / 65535) * (box[a + 3] - box[a]);
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(rgb, 3, true));
        return geometry;
      }));
    }
    return cache.get(url);
  }

  // The start view of a sample: the clouds are moved into the frame of the paper render (orbit centre at the
  // origin, the render's up as +y), and the camera looks at the origin from +z.
  let frame = new THREE.Matrix4(), dist = 1;
  function setView(view) {
    const position = new THREE.Vector3(...view.position), centre = new THREE.Vector3(...view.target);
    const back = position.clone().sub(centre);
    dist = back.length();
    back.normalize();
    const up = new THREE.Vector3(...view.up).normalize();
    const right = new THREE.Vector3().crossVectors(up, back).normalize();
    const rot = new THREE.Matrix4().makeBasis(right, up, back).transpose();
    frame = rot.multiply(new THREE.Matrix4().makeTranslation(-centre.x, -centre.y, -centre.z));
    camera.fov = view.fov;
    camera.near = dist / 1000;
    camera.far = dist * 100;
    camera.updateProjectionMatrix();
    material.uniforms.uFoot.value = FOOT / view.f0;
    reset();
  }

  function reset() {
    target.set(0, 0, 0);
    camera.up.set(0, 1, 0);
    camera.position.set(0, 0, dist);
    camera.lookAt(target);
    Object.values(views).forEach((v) => v.controls.update());
    render();
  }

  function show(key, url) {
    const v = views[key], panel = $('panel-' + key);
    panel.classList.add('is-loading');
    v.url = url;
    load(url).then((geometry) => {
      if (v.url !== url) return;      // a newer request replaced this one
      v.scene.clear();
      const points = new THREE.Points(geometry, material);
      points.matrixAutoUpdate = false;
      points.matrix.copy(frame);
      v.scene.add(points);
      panel.classList.remove('is-loading');
      render();
    }).catch((e) => {
      console.error(e);
      panel.classList.remove('is-loading');
    });
  }

  return { setView, show };
}

// ---------- page controls ----------

function format(v) {
  if (v === null || v === undefined) return '';
  const digits = state.task === 'normal' || state.task === 'instance' ? 1 : 2;
  return task().metric + ' ' + v.toFixed(digits) + task().unit;
}

function best(values) {
  const t = task();
  let out = null;
  t.methods.forEach((m) => {
    const v = values[m];
    if (v === null || v === undefined) return;
    if (out === null || (t.lower ? v < values[out] : v > values[out])) out = m;
  });
  return out;
}

function pills(box, items, active, onClick) {
  box.innerHTML = '';
  items.forEach(([key, text]) => {
    const b = document.createElement('button');
    b.textContent = text;
    b.className = key === active ? 'is-active' : '';
    b.addEventListener('click', () => onClick(key));
    box.appendChild(b);
  });
}

function setTask(name) {
  const t = DATA.tasks[name];
  state.task = name;
  state.sample = 0;
  state.other = t.methods.includes(DEFAULT[name]) ? DEFAULT[name] : t.methods[0];
  state.mode = t.clouds && viewer ? '3d' : 'pred';
  pills($('compare-tasks'), ORDER.map((k) => [k, DATA.tasks[k].title]), name, setTask);
  const select = $('compare-select');
  select.innerHTML = '';
  t.methods.filter((m) => m !== DATA.ours).forEach((m) => {
    const o = document.createElement('option');
    o.value = m;
    o.textContent = label(m);
    select.appendChild(o);
  });
  select.value = state.other;
  const thumbs = $('compare-thumbs');
  thumbs.innerHTML = '';
  // two even rows when there are more than six samples: 10 as 5 + 5, 12 as 6 + 6
  const n = t.samples.length;
  thumbs.style.gridTemplateColumns = 'repeat(' + (n > 6 ? Math.ceil(n / 2) : n) + ', auto)';
  t.samples.forEach((s, k) => {
    const img = document.createElement('img');
    img.src = file(s, 'thumb');
    img.alt = s.dataset + ' sample ' + (k + 1);
    img.title = s.dataset;
    img.addEventListener('click', () => { state.sample = k; render(true); });
    thumbs.appendChild(img);
  });
  render(true);
}

function modes() {
  const t = task(), out = [];
  if (t.clouds && viewer) out.push(['3d', '3D']);
  if (!t.clouds || !viewer) out.push(['pred', 'Prediction']);
  if (t.errors) out.push(['error', 'Error']);
  return out;
}

function render(newSample) {
  const s = sample(), top = best(s.values);
  const is3d = state.mode === '3d', error = state.mode === 'error';
  if (is3d && newSample) viewer.setView(s.view);
  Object.keys(PANELS).forEach((key) => {
    const panel = $('panel-' + key), m = method(key);
    panel.classList.toggle('is-3d', is3d);
    if (is3d) {
      viewer.show(key, file(s, m, '.bin'));
    } else {
      panel.classList.remove('is-loading');
      // in the error view the left panel shows the input image: the error maps live in its pixels
      const name = error && key === 'gt' ? 'Image' : m + (error ? '_error' : '');
      panel.querySelector('img').src = file(s, name);
    }
  });
  $('tag-gt').textContent = error ? 'Image' : 'Ground truth';
  $('tag-ours').innerHTML = '<b>' + label(DATA.ours) + ' (ours)</b> <span' +
    (top === DATA.ours ? ' class="is-best"' : '') + '>' + format(s.values[DATA.ours]) + '</span>';
  $('compare-value').innerHTML = format(s.values[state.other]);
  const ms = modes();
  $('compare-mode').style.display = ms.length > 1 ? '' : 'none';
  pills($('compare-mode'), ms, state.mode, (m) => {
    const was3d = state.mode === '3d';
    state.mode = m;
    render(m === '3d' && !was3d);
  });
  document.querySelectorAll('#compare-thumbs img').forEach((img, k) => img.classList.toggle('is-active', k === state.sample));
  const note = error ? ERROR_NOTE : (is3d || state.task !== 'point' ? NOTE[state.task] || '' : '');
  $('compare-note').textContent = s.dataset + '. ' + note;
}

if (DATA && $('compare-select')) {
  try {
    viewer = makeViewer();
  } catch (e) {
    console.error('3D viewer unavailable, showing the renders instead', e);
    viewer = null;
  }
  $('compare-select').addEventListener('change', function () { state.other = this.value; render(false); });
  setTask('point');
}
