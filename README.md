# Multiscale Viewer

Interactive, multi-scale deep zoom viewer for the department camera array, registered into a unified coordinate frame.

**Live Interactive Web Viewers (GitHub Pages):**
- **Department Photo Viewer:** [https://gord123098.github.io/department-photo/](https://gord123098.github.io/department-photo/)
- **Multi-Scale Video Viewer:** [https://gord123098.github.io/department-photo-video/](https://gord123098.github.io/department-photo-video/)

**Repositories (Auto-Synchronized):**
- [https://github.com/Gord123098/department-photo](https://github.com/Gord123098/department-photo)
- [https://github.com/arizonaCameraLab/department-photo](https://github.com/arizonaCameraLab/department-photo)

---

## Multi-Scale Demonstration

![Multiscale Viewer Layer Footprints](assets/multiscale_layers_overview.jpg)
*Multi-scale registration showing the wide color context frame (8FF9), 12 mm stitched tier footprint, and high-resolution close detail (A + B) telephoto array over the crowd.*

### Telephoto Sampling Boundary
![Telephoto Resolution Transition](assets/telephoto_resolution_transition.jpg)
*Deep zoom crop (15.41×) capturing the transition boundary where the high-resolution monochrome telephoto array has sampled vs. the outer wide context.*

---

## Key Features

1. **Multi-Scale Pyramid (1.00× to 4.80×)**
   - **Wide Panorama Base (1.00×):** 8FF9 color reference camera covering the entire scene.
   - **12 mm Stitched Tier (1.55×):** Blended panoramic array from monochrome cameras `6450` and `643D`.
   - **Close Detail Stitched Tier (4.80×):** Unified 6-camera telephoto array (`447`, `6453`, `6445`, `645C`, `458`, `644F`) providing high-resolution detail.
   - **All-Monochrome Stitched Array:** Seamless 8-camera mosaic.
   - **25 Deep Zoom Layers:** All individual sensors and composite tiers independently navigable.

2. **Highest-Resolution Color Fusion Pipeline**
   - Incorporates auxiliary optical color reference `642F` (1.664× optical scale, 2.77× pixel density) with edge-preserving affine warp and DIS dense optical flow.
   - Feathered Voronoi seam blending across multi-camera overlaps to eliminate abrupt color boundaries.
   - Instant toggle between **Color Fused** and **Native Monochrome** (press <kbd>M</kbd>).

3. **Layer Compositor (Stack)**
   - Arbitrary layer stacking order: place any layer above or below any other layer with `[▲]` / `[▼]` reordering.
   - Checkbox visibility toggles for each layer.
   - Individual 0%–100% opacity sliders for smooth dissolve, alignment inspection, and comparison.
   - **Quick Comparison Presets:**
     - *High-Res over Low-Res:* Stacks Close Detail (4.80×) on top of 12 mm Stitched (1.55×).
     - *Low-Res over High-Res:* Inverts the stack with 75% opacity to see through to underlying detail.
     - *Mono over Fused:* Compares native monochrome sharpness directly over color fusion.
     - *Auto Zoom Mode:* Re-engages scroll-based focal tier transitions.

4. **Telephoto Field Edge Navigation**
   - Instant navigation buttons (`[Left]`, `[Right]`, `[Top]`, `[Bottom]`, `[Center]`) to zoom directly to the spatial boundaries of the telephoto array against the surrounding 12 mm field.
   - Keyboard shortcut <kbd>E</kbd> cycles through the perimeter edges.

5. **Full Mobile & Desktop Support**
   - Responsive touch gestures: pinch-to-zoom, pan, double-tap zoom.
   - Clean, lightweight, self-contained static web application hosted on GitHub Pages.
   - Focused panoramic viewport with intuitive left-hand sidebar controls.

---

## Synchronized Multi-Scale Video Viewer

A synchronized multi-scale video extension of this camera array across **153 continuous frames at native 24 fps** is available with interactive multi-tier deep zoom playback:

- **Live Video Viewer:** [https://gord123098.github.io/department-photo-video/](https://gord123098.github.io/department-photo-video/)
- **Video Repositories:** [https://github.com/Gord123098/department-photo-video](https://github.com/Gord123098/department-photo-video) · [https://github.com/arizonaCameraLab/department-photo-video](https://github.com/arizonaCameraLab/department-photo-video)

### Highlights & Architecture
- **153 Successive Synchronized Frames:** Every high-resolution sensor advances by exactly 1 frame per step (spanning 6.38 seconds at cinema-standard 24 fps).
- **All-Intra Keyframe Seeking (`-g 1`):** Every frame across all video streams is an independent IDR keyframe, enabling zero-latency, frame-accurate timeline scrubbing and frame-by-frame stepping in the browser.
- **Identical Registration & Fusion Pipeline:** Incorporates ALIKED + LightGlue + USAC_MAGSAC homographies, Lanczos-4 projective warping, DIS optical flow parallax refinement, and bilateral-filtered YCrCb pan-sharpened color fusion on every frame.
- **Four Concurrent Video Tiers:**
  - **Wide Base:** 2096×1600 color context panorama.
  - **12 mm Stitched Tier:** 3072×1552 blended wide-telephoto tier.
  - **Close Detail Array:** 3840×1712 4K color-fused stream.
  - **Native Mono Array:** 3840×1712 4K monochrome stream.
- **Local Playback & Streaming:** Powered by a lightweight Python HTTP range-request server (`serve_viewer.py`) supporting HTTP 206 Partial Content byte-range seeks at `http://127.0.0.1:8765/`.

---

## Credits & Acknowledgments

- **Heterogeneous Camera Array**: Based on the processing pipeline and camera registration architecture from the [Arizona Camera Lab](https://github.com/arizonaCameraLab).
- **Upstream Repository**: [arizonaCameraLab/Heterogeneous-Camera-Array](https://github.com/arizonaCameraLab/Heterogeneous-Camera-Array)
- **Author & Original Viewer**: Special credit to **Adel** ([@adel111700](https://github.com/adel111700)) for authoring the heterogeneous camera array pipeline and original viewer.

---

## Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| <kbd>[</kbd> / <kbd>]</kbd> | Put away / Toggle Sidebar Menu |
| <kbd>E</kbd> | Cycle through Telephoto Field Edges (`Left` → `Right` → `Top` → `Bottom` → `Center`) |
| <kbd>M</kbd> / <kbd>F</kbd> | Toggle representation (**Color Fused** ⇄ **Native Mono**) |
| <kbd>0</kbd> | Auto Zoom Mode (focal tier switches dynamically on scroll) |
| <kbd>1</kbd> | Wide Color Panorama (8FF9) |
| <kbd>2</kbd> | 12 mm Stitched Array (6450 + 643D) |
| <kbd>3</kbd> | Close Detail A + B Stitched Array (6 cameras) |
| <kbd>4</kbd> | All-Monochrome Array |
| <kbd>Space</kbd> | Reset viewport to full panoramic frame |

---

## Local Development & Offline Viewing

```bash
# Serve locally
python3 serve_viewer.py

# Open in browser
http://127.0.0.1:8765/
```
