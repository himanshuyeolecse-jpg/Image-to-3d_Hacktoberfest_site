# Wukong relief site

Static site. No build step, no server code.

## Run locally
    python3 -m http.server 8000      # then open http://localhost:8000
Opening index.html from disk will not work (browsers block module and model loading over file://).

## Deploy
- Render: new Static Site from this folder (render.yaml included). Publish directory: `.`
- Custom domain (.tech): add it in Render's Custom Domains, then set the DNS records Render shows.

## Fully offline depth model (optional)
By default the depth model downloads from Hugging Face on first use and the browser caches it.
To bundle it instead, download `onnx-community/depth-anything-v2-small` and place it at:
    models/onnx-community/depth-anything-v2-small/   (config.json, preprocessor_config.json, onnx/…)
The page checks `models/` first.

## Structure
- index.html: intro film, handoff to the 3D Wukong model, scroll sections
- tool.js, tool-core.js: photo to solid 3D object (finds the object, mirrors the back, .glb and .stl export)
- assets/: film, music, wukong.glb (3D model, 75 MB, used as supplied and not recompressed), sample photos, u2netp.onnx (object finder)
- showcase/: sample models (see credits)
- vendor/: three.js (MIT), transformers.js (Apache-2.0) with onnxruntime-web (MIT), opencv.js (Apache-2.0)

## Credits and licences
Sample models, from Khronos glTF Sample Assets:
- Fox: model by PixelMannen (CC0); rigging and animation by tomkranis (CC BY 4.0); glTF conversion by @AsoboStudio and @scurest (CC BY 4.0)
- Toy car: Guido Odendahl (CC0)
- Lantern: Microsoft and sbtron (CC0)
- Avocado: Microsoft (CC0)
- assets/sample_*.jpg: renders of the Fox, Avocado and Lantern models above on a plain background (derived from CC0 and CC BY works; credit as above)
Object finder: U²-Net small (u2netp.onnx), Apache-2.0, from the rembg project releases; runs with onnxruntime-web (MIT). It ships in assets/ so it works offline.
Depth model: Depth Anything V2 Small (Apache-2.0) via onnx-community.
Wukong figure: fan work inspired by Black Myth: Wukong (Game Science). assets/wukong.glb was generated with Tripo from reference images of a commercial statue design; credit the character and original sculptors, and replace the model with your own or licensed art before any commercial release.
Hosting note: wukong.glb is 75 MB. GitHub warns above 50 MB and rejects files over 100 MB; use Git LFS if a push is refused.
