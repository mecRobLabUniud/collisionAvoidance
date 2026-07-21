# Panda DAE Viewer

A three.js viewer that renders the **real Franka Emika Panda visual meshes**
(`.dae`, from `franka_ros`'s `franka_description` package) driven by exact
modified-DH forward kinematics — same parameter table Franka publishes, and
the same joint-frame layout used in the URDF/xacro.

## Why this isn't a single HTML file

The visual meshes total ~9 MB across `link0`–`link7`, `hand`, and `finger`,
each with several materials per part. That's too large to inline as base64
in one file, and `THREE.ColladaLoader` needs real `fetch`/XHR calls to load
`.dae` files — which browsers block from `file://` pages. So this ships as a
small local project instead of a single portable artifact.

## Running it

Meshes are already bundled in `meshes/visual/` (copied straight from
`franka_ros/franka_description/meshes/visual/`), so there's nothing to
download or symlink. From this folder:

```bash
python3 -m http.server 8000
```

then open `http://localhost:8000` in a browser. (Any static file server
works — `npx serve`, VS Code's Live Server, etc. Just not `file://`.)

## What's real vs. approximated

- `link0`–`link7`, `hand`: the actual `franka_description` `.dae` visual
  meshes, loaded with `THREE.ColladaLoader`, materials and all (the real
  pearl-white body / black joint housings come straight from the files).
- Joint frames and the hand's -45° flange twist: computed from Franka's
  official modified-DH table, cross-checked against the joint `origin`
  values in `franka_arm.xacro` / `franka_hand.xacro`.
- Fingers: `franka_description` only ships one `finger.dae` — the real
  robot reuses it for both sides (the right one mirrored 180° about Z, per
  `franka_hand.xacro`), which is what this does too. Gripper slider drives
  both along the real prismatic joint axis and offset (`0.0584 m` from the
  hand origin, 0–40 mm travel each).

## Files

```
index.html            page shell + control panel
js/app.js              kinematics + mesh loading/wiring
vendor/ColladaLoader.js three.js r128's official (unmodified) Collada loader
meshes/visual/*.dae     Franka's real visual meshes
```

## Swapping in your own workspace's meshes

If you'd rather point at the copy already in your catkin workspace instead
of the bundled one, replace `meshes/visual/` with a symlink:

```bash
rm -rf meshes/visual
ln -s ~/Desktop/franka_emika_ws/src/franka_ros/franka_description/meshes/visual meshes/visual
```
