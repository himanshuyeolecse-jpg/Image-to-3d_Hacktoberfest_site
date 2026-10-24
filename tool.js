import * as THREE from 'three';
import {OrbitControls} from './vendor/controls/OrbitControls.js';
import {GLTFExporter} from './vendor/exporters/GLTFExporter.js';
import {RoomEnvironment} from './vendor/env/RoomEnvironment.js';
import {fitSize,smooth,normalize,quickDepth,colourDistance,otsu,cleanMask,buildSolid,openEdges,solidToSTL} from './tool-core.js';
const $=id=>document.getElementById(id), msg=t=>$('tmsg').textContent=t;
const SAMPLES=['assets/sample_fox.jpg','assets/sample_avocado.jpg','assets/sample_lantern.jpg']; let si=0;
const st={load:null,find:null,depth:null,mesh:null,exp:null,tris:null,grid:null,mode:null,how:null,closed:null};
let R,scene,cam,ctl,mesh,tex,nx,ny,depth0,mask0,solid,vis=false,busy=false;
function showStats(){const f=v=>v==null?'–':v<1?'<1 ms':v>=1000?(v/1000).toFixed(1)+' s':Math.round(v)+' ms';
  $('tstat').innerHTML=`<tr><td>Depth model download and start</td><td>${st.load==null?'cached or not used':f(st.load)}</td></tr><tr><td>Depth map (${st.mode||'–'})</td><td>${f(st.depth)}</td></tr><tr><td>Finding the object (${st.how||'–'})</td><td>${f(st.find)}</td></tr><tr><td>Building the solid</td><td>${f(st.mesh)}</td></tr><tr><td>Last export</td><td>${f(st.exp)}</td></tr><tr><td>Grid and triangles</td><td>${st.grid||'–'} / ${st.tris||'–'}</td></tr><tr><td>Printable check</td><td>${st.closed==null?'–':st.closed===0?'closed shape':st.closed+' open edges (slicers repair this)'}</td></tr>`}
function net(){$('tnet').textContent=navigator.onLine?'online':'offline, still working'}
addEventListener('online',net);addEventListener('offline',net);net();showStats();
function boot(){
  R=new THREE.WebGLRenderer({canvas:$('tcv'),antialias:true,preserveDrawingBuffer:true}); R.setPixelRatio(Math.min(devicePixelRatio,2)); R.toneMapping=THREE.ACESFilmicToneMapping; R.setClearColor(0x0e1722);
  scene=new THREE.Scene(); scene.environment=new THREE.PMREMGenerator(R).fromScene(new RoomEnvironment(),.04).texture; scene.environmentIntensity=.7;
  const k=new THREE.DirectionalLight(0xffffff,2.2); k.position.set(-2,3,5); scene.add(k); const k2=new THREE.DirectionalLight(0xdfe9ff,1.6); k2.position.set(2,2,-5); scene.add(k2);
  cam=new THREE.PerspectiveCamera(35,4/3,.1,100); cam.position.set(2.2,1.2,8);
  ctl=new OrbitControls(cam,$('tcv')); ctl.enableDamping=true; ctl.enablePan=false; ctl.minDistance=4; ctl.maxDistance=14; ctl.autoRotate=true; ctl.autoRotateSpeed=1.4;
  ctl.addEventListener('start',()=>ctl.autoRotate=false);
  new ResizeObserver(()=>{const b=$('tbox'),w=b.clientWidth,h=b.clientHeight;R.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix()}).observe($('tbox'));
  (function loop(){requestAnimationFrame(loop); if(vis){ctl.update();R.render(scene,cam)}})();
}
new IntersectionObserver(e=>{vis=e[0].isIntersecting; if(vis&&!R)boot()},{rootMargin:'300px'}).observe($('tool'));
// ---- depth model (Depth Anything V2 Small, runs in the browser) ----
let est=null;
async function estimator(){
  if(est) return est;
  est=(async()=>{
    const tf=await import('./vendor/tf/transformers.web.min.js'), t=performance.now();
    tf.env.allowLocalModels=true; tf.env.localModelPath=new URL('./models/',location.href).href; tf.env.backends.onnx.wasm.wasmPaths=new URL('./vendor/tf/',location.href).href;
    const cb=p=>{if(p.status==='progress')msg(`Downloading depth model… ${Math.round(p.progress)}% (${String(p.file).split('/').pop()})`)};
    const id='onnx-community/depth-anything-v2-small'; let p;
    try{p=await tf.pipeline('depth-estimation',id,{dtype:'q8',progress_callback:cb})}catch{p=await tf.pipeline('depth-estimation',id,{progress_callback:cb})}
    st.load=performance.now()-t; return p})();
  est.catch(()=>est=null); return est;
}
async function aiDepth(canvas){
  const p=await estimator(); msg('Estimating depth…');
  const blob=await new Promise(r=>canvas.toBlob(r,'image/png')), url=URL.createObjectURL(blob);
  const out=await p(url); URL.revokeObjectURL(url); const im=out.depth, c=document.createElement('canvas'); c.width=im.width; c.height=im.height;
  const id=new ImageData(im.width,im.height), ch=im.channels||1; for(let i=0;i<im.width*im.height;i++){const v=im.data[i*ch];id.data.set([v,v,v,255],i*4)}
  c.getContext('2d').putImageData(id,0,0); const g=document.createElement('canvas'); g.width=nx; g.height=ny; g.getContext('2d').drawImage(c,0,0,nx,ny);
  const px=g.getContext('2d').getImageData(0,0,nx,ny).data, d=new Float32Array(nx*ny); for(let i=0;i<d.length;i++)d[i]=px[i*4]/255; return d;
}
// ---- object finder: U²-Net (small), a saliency model that runs here with onnxruntime-web ----
let segp=null;
function loadSeg(){ if(segp) return segp;
  segp=(async()=>{const ort=await import('./vendor/ort/ort.wasm.min.mjs'); ort.env.wasm.wasmPaths=new URL('./vendor/ort/',location.href).href; ort.env.wasm.numThreads=1;
    const s=await ort.InferenceSession.create(new URL('assets/u2netp.onnx',location.href).href,{executionProviders:['wasm']}); return {ort,s}})();
  segp.catch(()=>segp=null); return segp }
async function saliency(src){
  const {ort,s}=await loadSeg(), S=320, c=document.createElement('canvas'); c.width=c.height=S; const x=c.getContext('2d'); x.drawImage(src,0,0,S,S);
  const px=x.getImageData(0,0,S,S).data, f=new Float32Array(3*S*S), mean=[.485,.456,.406], sd=[.229,.224,.225];
  for(let i=0;i<S*S;i++)for(let k=0;k<3;k++)f[k*S*S+i]=(px[i*4+k]/255-mean[k])/sd[k];
  const out=await s.run({[s.inputNames[0]]:new ort.Tensor('float32',f,[1,3,S,S])}), o=out[s.outputNames[0]].data;
  let lo=1e9,hi=-1e9; for(let i=0;i<S*S;i++){if(o[i]<lo)lo=o[i];if(o[i]>hi)hi=o[i]}
  const g=x.createImageData(S,S); for(let i=0;i<S*S;i++){const v=Math.round(255*(o[i]-lo)/Math.max(hi-lo,1e-6));g.data.set([v,v,v,255],i*4)}
  x.putImageData(g,0,0); const r=document.createElement('canvas'); r.width=nx; r.height=ny; const rx=r.getContext('2d'); rx.imageSmoothingQuality='high'; rx.drawImage(c,0,0,nx,ny);
  const rp=rx.getImageData(0,0,nx,ny).data, p=new Float32Array(nx*ny); for(let i=0;i<p.length;i++)p[i]=rp[i*4]/255; return p }
// ---- find the object: depth + colour first, then GrabCut (OpenCV) to sharpen the outline ----
let cvp=null;
// OpenCV.js can be a promise-like object that resolves to itself; awaiting it directly loops forever, so we wait without returning it.
function loadCV(){ if(cvp) return cvp;
  cvp=new Promise((ok,no)=>{const s=document.createElement('script'); s.src='vendor/cv/opencv.js'; s.onerror=()=>no(new Error('opencv failed to load'));
    s.onload=()=>{let done=false; const fin=m=>{if(!done&&m&&m.Mat){done=true;ok({cv:m})}}, c=window.cv;
      if(c&&typeof c.then==='function')c.then(m=>{fin(m)});
      if(c&&!c.Mat)c.onRuntimeInitialized=()=>fin(window.cv);
      const poll=()=>{if(done)return; if(window.cv&&window.cv.Mat)fin(window.cv); else setTimeout(poll,80)}; poll();
      setTimeout(()=>{if(!done)no(new Error('opencv timeout'))},45000)}; document.head.append(s)});
  cvp.catch(()=>{}); return cvp }
function grabcut(cv,rgba,init){
  const src=cv.matFromImageData(new ImageData(new Uint8ClampedArray(rgba),nx,ny)), rgb=new cv.Mat(); cv.cvtColor(src,rgb,cv.COLOR_RGBA2RGB);
  const sure=new Uint8Array(init.length), near=new Uint8Array(init.length); // sure foreground (eroded) and "possibly" zone (dilated)
  const er=(m,g)=>{const o=new Uint8Array(m.length);for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){let a=g?0:1;for(let j=-2;j<=2;j++)for(let i=-2;i<=2;i++){const xx=x+i,yy=y+j,v=xx>=0&&yy>=0&&xx<nx&&yy<ny?m[yy*nx+xx]:0;g?(v&&(a=1)):(!v&&(a=0))}o[y*nx+x]=a}return o};
  const E=er(init,0),D=er(init,1), mk=new cv.Mat(ny,nx,cv.CV_8UC1);
  for(let i=0;i<init.length;i++)mk.data[i]=E[i]?cv.GC_FGD:init[i]?cv.GC_PR_FGD:D[i]?cv.GC_PR_BGD:cv.GC_BGD;
  const b=new cv.Mat(),f=new cv.Mat(); cv.grabCut(rgb,mk,new cv.Rect(0,0,nx,ny),b,f,4,cv.GC_INIT_WITH_MASK);
  const out=new Uint8Array(init.length); for(let i=0;i<out.length;i++)out[i]=(mk.data[i]===cv.GC_FGD||mk.data[i]===cv.GC_PR_FGD)?1:0;
  [src,rgb,mk,b,f].forEach(m=>m.delete()); return out }
async function findObject(rgba,depth,src){
  const t=performance.now(), N=nx*ny, frac=m=>m.reduce((a,b)=>a+b,0)/N, cd=normalize(colourDistance(rgba,nx,ny));
  const mkMask=s=>{const th=otsu(s),m=new Uint8Array(N);for(let i=0;i<N;i++)m[i]=s[i]>th?1:0;return cleanMask(m,nx,ny)};
  let how='', m=null;
  try{msg('Finding the object…');const p=smooth(await saliency(src),nx,ny,1),mm=new Uint8Array(N);for(let i=0;i<N;i++)mm[i]=p[i]>.5?1:0;m=cleanMask(mm,nx,ny);how='U²-Net'}catch(e){console.warn('U²-Net unavailable',e);m=null}
  if(!m||frac(m)<.03||frac(m)>.95){m=null;
    if(depth){const dn=normalize(depth),s=new Float32Array(N);for(let i=0;i<N;i++)s[i]=.5*dn[i]+.5*cd[i];m=mkMask(s);how='depth + colour'}
    if(!m||frac(m)<.04||frac(m)>.92){m=mkMask(cd);how='colour'}}
  if(frac(m)<.03||frac(m)>.95){m=new Uint8Array(N);const g=Math.round(Math.min(nx,ny)*.04);for(let y=g;y<ny-g;y++)for(let x=g;x<nx-g;x++)m[y*nx+x]=1;how='whole photo (no clear object found)'}
  else if($('trefine').checked){try{msg('Sharpening the outline…');const {cv}=await loadCV(),g=cleanMask(grabcut(cv,rgba,m),nx,ny),a=frac(g),b=frac(m);if(a>.7*b&&a<1.3*b){m=g;how+=' + GrabCut'}}catch(e){console.warn('GrabCut skipped',e)}}
  st.find=performance.now()-t; st.how=how; return m }
// ---- build and show ----
function rebuild(){
  const t=performance.now();
  solid=buildSolid(depth0,mask0,nx,ny,{thick:+$('trelief').value,detail:+$('tdetail').value*(st.mode==='AI depth'?1:.35),mirror:$('tmirror').checked,width:4});
  if(mesh){scene.remove(mesh);mesh.geometry.dispose()}
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(solid.positions,3)); g.setAttribute('uv',new THREE.BufferAttribute(solid.uvs,2)); g.setIndex(new THREE.BufferAttribute(solid.indices,1)); g.computeVertexNormals();
  mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({map:tex,roughness:.8,metalness:0,side:THREE.DoubleSide})); scene.add(mesh);
  st.mesh=performance.now()-t; st.tris=solid.indices.length/3; st.closed=openEdges(solid.positions,solid.indices); showStats();
}
async function process(canvas){
  if(busy) return; busy=true; if(!R)boot(); $('tglb').disabled=$('tstl').disabled=true;
  try{
    const g=fitSize(canvas.width,canvas.height,256); nx=g.w; ny=g.h; st.grid=`${nx}×${ny}`; st.load=st.load;
    const tc=document.createElement('canvas'), s=fitSize(canvas.width,canvas.height,768); tc.width=s.w; tc.height=s.h; tc.getContext('2d').drawImage(canvas,0,0,s.w,s.h);
    tex=new THREE.CanvasTexture(tc); tex.colorSpace=THREE.SRGBColorSpace; tex.anisotropy=4;
    const sm=document.createElement('canvas'); sm.width=nx; sm.height=ny; const sx=sm.getContext('2d'); sx.drawImage(canvas,0,0,nx,ny); const rgba=sx.getImageData(0,0,nx,ny).data;
    const t=performance.now(); let d=null, aiFailed=false; st.mode='AI depth';
    if($('tmode').value==='ai'){try{d=await aiDepth(tc)}catch(e){console.error(e); aiFailed=true; d=null}}
    if(!d){st.mode='quick depth'; d=quickDepth(rgba,nx,ny)} st.depth=performance.now()-t;
    mask0=await findObject(rgba,st.mode==='AI depth'?smooth(d,nx,ny,1):null,tc);
    depth0=normalize(smooth(d,nx,ny,1),mask0); rebuild();
    msg(st.mode==='AI depth'?`Done. Found the object (${st.how}). Drag to spin it all the way round.`:aiFailed?`Done, but the AI depth model could not load (it needs one online run, or a copy in /models), so this used quick depth.`:`Done with quick depth (${st.how}). Pick “AI depth” for a more accurate shape.`);
    $('tglb').disabled=$('tstl').disabled=false;
  }catch(e){console.error(e); msg('Could not read that image. Try a JPG or PNG.')} finally{busy=false}
}
async function fromBlob(b){const bmp=await createImageBitmap(b), c=document.createElement('canvas'); c.width=bmp.width; c.height=bmp.height; c.getContext('2d').drawImage(bmp,0,0); return process(c)}
$('tfile').onchange=e=>e.target.files[0]&&fromBlob(e.target.files[0]);
$('tsample').onclick=async()=>{const u=SAMPLES[si++%SAMPLES.length]; fromBlob(await (await fetch(u)).blob())};
const dz=$('tdrop'); ['dragover','dragenter'].forEach(n=>dz.addEventListener(n,e=>{e.preventDefault();dz.classList.add('over')})); ['dragleave','drop'].forEach(n=>dz.addEventListener(n,()=>dz.classList.remove('over')));
dz.addEventListener('drop',e=>{e.preventDefault(); const f=e.dataTransfer.files[0]; f&&f.type.startsWith('image/')?fromBlob(f):msg('Drop an image file (JPG or PNG).')});
addEventListener('paste',e=>{const f=[...(e.clipboardData?.files||[])].find(f=>f.type.startsWith('image/')); f&&fromBlob(f)});
['trelief','tdetail'].forEach(i=>$(i).oninput=()=>depth0&&rebuild()); $('tmirror').onchange=()=>depth0&&rebuild();
function save(data,name,type){const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([data],{type})); a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),4000)}
$('tglb').onclick=()=>{const t=performance.now(); new GLTFExporter().parse(mesh,r=>{save(r,'object.glb','model/gltf-binary'); st.exp=performance.now()-t; showStats()},e=>msg('GLB export failed.'),{binary:true})};
$('tstl').onclick=()=>{const t=performance.now(); save(solidToSTL(solid,80),'object.stl','model/stl'); st.exp=performance.now()-t; showStats()};
window.__tool={process,view:(x,y,z)=>{ctl.autoRotate=false;cam.position.set(x,y,z);ctl.update()},get st(){return st},get solid(){return solid}};
