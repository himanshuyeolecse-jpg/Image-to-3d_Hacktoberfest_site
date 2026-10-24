// Pure functions (no DOM) so they can be tested in Node.
export function fitSize(w,h,max){const k=Math.min(1,max/Math.max(w,h));return{w:Math.max(2,Math.round(w*k)),h:Math.max(2,Math.round(h*k))}}
function boxBlur(a,nx,ny,r){const t=new Float32Array(a.length),o=new Float32Array(a.length);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){let s=0,n=0;for(let k=-r;k<=r;k++){const xx=x+k;if(xx>=0&&xx<nx){s+=a[y*nx+xx];n++}}t[y*nx+x]=s/n}
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){let s=0,n=0;for(let k=-r;k<=r;k++){const yy=y+k;if(yy>=0&&yy<ny){s+=t[yy*nx+x];n++}}o[y*nx+x]=s/n}
  return o}
export const smooth=(d,nx,ny,r=1)=>boxBlur(d,nx,ny,r);
export function normalize(d,mask){const v=[];for(let i=0;i<d.length;i++)if(!mask||mask[i])v.push(d[i]);if(!v.length)return d;
  v.sort((a,b)=>a-b);const lo=v[Math.floor(v.length*.02)],hi=v[Math.floor(v.length*.98)],k=1/Math.max(hi-lo,1e-6);
  const o=new Float32Array(d.length);for(let i=0;i<d.length;i++)o[i]=Math.min(1,Math.max(0,(d[i]-lo)*k));return o}
export function otsu(a){const h=new Float64Array(256);for(const v of a)h[Math.min(255,Math.max(0,Math.round(v*255)))]++;
  const n=a.length;let sum=0;for(let i=0;i<256;i++)sum+=i*h[i];let sb=0,wb=0,best=0,th=128;
  for(let i=0;i<256;i++){wb+=h[i];if(!wb)continue;const wf=n-wb;if(!wf)break;sb+=i*h[i];const mb=sb/wb,mf=(sum-sb)/wf,v=wb*wf*(mb-mf)**2;if(v>best){best=v;th=i}}
  return th/255}
// Quick relief with no AI: brighter and more central pixels stand out. Works instantly and offline.
export function quickDepth(rgba,nx,ny){const L=new Float32Array(nx*ny);
  for(let i=0;i<L.length;i++)L[i]=(.3*rgba[i*4]+.59*rgba[i*4+1]+.11*rgba[i*4+2])/255;
  const base=boxBlur(L,nx,ny,Math.max(2,Math.round(nx/60))),d=new Float32Array(L.length);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const cx=(x/nx-.5)*2,cy=(y/ny-.5)*2,c=1-Math.min(1,Math.hypot(cx,cy)/1.2);d[y*nx+x]=.65*base[y*nx+x]+.35*c}
  return normalize(d)}
// How different each pixel is from the background colour (median of the photo's border), scaled 0..1.
export function colourDistance(rgba,nx,ny){const b=[[],[],[]],m=Math.max(2,Math.round(Math.min(nx,ny)*.03));
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)if(x<m||y<m||x>=nx-m||y>=ny-m){const i=(y*nx+x)*4;b[0].push(rgba[i]);b[1].push(rgba[i+1]);b[2].push(rgba[i+2])}
  const med=c=>{c.sort((p,q)=>p-q);return c[c.length>>1]},bg=[med(b[0]),med(b[1]),med(b[2])],d=new Float32Array(nx*ny);let mx=1e-6;
  for(let i=0;i<d.length;i++){d[i]=Math.hypot(rgba[i*4]-bg[0],rgba[i*4+1]-bg[1],rgba[i*4+2]-bg[2]);if(d[i]>mx)mx=d[i]}
  const s=boxBlur(d,nx,ny,1);for(let i=0;i<s.length;i++)s[i]/=mx;return s}
function label(mask,nx,ny){const lab=new Int32Array(mask.length),sizes=[0],st=[];let n=0;
  for(let s=0;s<mask.length;s++){if(!mask[s]||lab[s])continue;n++;let c=0;lab[s]=n;st.push(s);
    while(st.length){const p=st.pop();c++;const x=p%nx,y=(p/nx)|0;
      if(x>0&&mask[p-1]&&!lab[p-1]){lab[p-1]=n;st.push(p-1)}if(x<nx-1&&mask[p+1]&&!lab[p+1]){lab[p+1]=n;st.push(p+1)}
      if(y>0&&mask[p-nx]&&!lab[p-nx]){lab[p-nx]=n;st.push(p-nx)}if(y<ny-1&&mask[p+nx]&&!lab[p+nx]){lab[p+nx]=n;st.push(p+nx)}}
    sizes.push(c)}
  return{lab,sizes,n}}
function morph(mask,nx,ny,grow){const o=new Uint8Array(mask.length);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){let all=1,any=0;
    for(let j=-1;j<=1;j++)for(let i=-1;i<=1;i++){const xx=x+i,yy=y+j,v=xx>=0&&yy>=0&&xx<nx&&yy<ny?mask[yy*nx+xx]:0;if(v)any=1;else all=0}
    o[y*nx+x]=grow?any:all}
  return o}
// Keep the biggest blob, fill holes inside it, round off thin spurs and one-pixel pinches.
export function cleanMask(mask,nx,ny){let m=morph(morph(mask,nx,ny,0),nx,ny,1);
  const {lab,sizes,n}=label(m,nx,ny);if(!n)return new Uint8Array(mask.length);
  let best=1;for(let i=2;i<=n;i++)if(sizes[i]>sizes[best])best=i;
  m=new Uint8Array(mask.length);for(let i=0;i<m.length;i++)m[i]=lab[i]===best?1:0;
  const inv=new Uint8Array(m.length);for(let i=0;i<m.length;i++)inv[i]=m[i]?0:1;
  const L=label(inv,nx,ny),edge=new Set();for(let x=0;x<nx;x++){edge.add(L.lab[x]);edge.add(L.lab[(ny-1)*nx+x])}for(let y=0;y<ny;y++){edge.add(L.lab[y*nx]);edge.add(L.lab[y*nx+nx-1])}
  for(let i=0;i<m.length;i++)if(inv[i]&&!edge.has(L.lab[i]))m[i]=1;
  // remove diagonal-only contacts (they make non-printable pinch points)
  for(let y=0;y<ny-1;y++)for(let x=0;x<nx-1;x++){const a=m[y*nx+x],b=m[y*nx+x+1],c=m[(y+1)*nx+x],d=m[(y+1)*nx+x+1];
    if(a&&d&&!b&&!c)m[y*nx+x]=0;else if(b&&c&&!a&&!d)m[y*nx+x+1]=0}
  return m}
export function distance(mask,nx,ny){const INF=1e9,d=new Float32Array(mask.length);for(let i=0;i<d.length;i++)d[i]=mask[i]?INF:0;
  const at=(x,y)=>x<0||y<0||x>=nx||y>=ny?0:d[y*nx+x];
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const i=y*nx+x;if(!d[i])continue;d[i]=Math.min(d[i],at(x-1,y)+1,at(x,y-1)+1,at(x-1,y-1)+1.414,at(x+1,y-1)+1.414)}
  for(let y=ny-1;y>=0;y--)for(let x=nx-1;x>=0;x--){const i=y*nx+x;if(!d[i])continue;d[i]=Math.min(d[i],at(x+1,y)+1,at(x,y+1)+1,at(x+1,y+1)+1.414,at(x-1,y+1)+1.414)}
  return d}
// Solid object from one photo: front surface from depth, back surface mirrored (or flat), joined at the outline.
export function buildSolid(depth,mask,nx,ny,{thick=.6,detail=.5,mirror=true,width=4}={}){
  let x0=nx,x1=0,y0=ny,y1=0;for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)if(mask[y*nx+x]){if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y}
  const bw=x1-x0+1,bh=y1-y0+1,size=Math.max(bw,bh),cell=width/size,dt=distance(mask,nx,ny),R=Math.max(3,.42*Math.min(bw,bh)),Tmax=thick*.38*Math.min(bw,bh)*cell;
  const vid=new Int32Array(nx*ny).fill(-1),P=[],U=[],RIM=[];let n=0;
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const i=y*nx+x;if(!mask[i])continue;vid[i]=n++;
    let rim=0;for(let j=-1;j<=1&&!rim;j++)for(let k=-1;k<=1;k++){const xx=x+k,yy=y+j;if(xx<0||yy<0||xx>=nx||yy>=ny||!mask[yy*nx+xx]){rim=1;break}}
    const e=rim?0:Math.min(1,Math.max(0,(dt[i]-1)/R)),round=Math.sqrt(1-(1-e)*(1-e)),dd=depth[i]-.5,h=Tmax*round*(.8+detail*.9*dd);
    RIM.push(rim);P.push((x-(x0+x1)/2)*cell,((y0+y1)/2-y)*cell,Math.max(0,h));U.push(x/(nx-1),1-y/(ny-1))}
  const idx=[];for(let y=0;y<ny-1;y++)for(let x=0;x<nx-1;x++){const a=vid[y*nx+x],b=vid[y*nx+x+1],c=vid[(y+1)*nx+x],d=vid[(y+1)*nx+x+1];if(a<0||b<0||c<0||d<0)continue;
    const tri=(p,q,r)=>{if(RIM[p]&&RIM[q]&&RIM[r])return;idx.push(p,q,r);idx.push(n+p,n+r,n+q)};tri(a,c,b);tri(b,c,d)}
  const pos=new Float32Array(n*6),uv=new Float32Array(n*4);
  for(let k=0;k<n;k++){const z=P[k*3+2];pos.set([P[k*3],P[k*3+1],z],k*3);pos.set([P[k*3],P[k*3+1],mirror?-z:0],(n+k)*3);uv.set([U[k*2],U[k*2+1]],k*2);uv.set([U[k*2],U[k*2+1]],(n+k)*2)}
  return{positions:pos,uvs:uv,indices:new Uint32Array(idx),count:n,size:width,cell}}
// Count edges that are not shared by exactly two triangles in opposite directions (0 = watertight).
export function openEdges(pos,idx){const key=i=>pos[i*3].toFixed(4)+','+pos[i*3+1].toFixed(4)+','+pos[i*3+2].toFixed(4),M=new Map();
  for(let t=0;t<idx.length;t+=3)for(let k=0;k<3;k++){const a=key(idx[t+k]),b=key(idx[t+(k+1)%3]);if(a===b)continue;M.set(a+'>'+b,(M.get(a+'>'+b)||0)+1)}
  let bad=0;for(const [e,c] of M){const [a,b]=e.split('>');if(c!==1||M.get(b+'>'+a)!==1)bad++}return bad}
export function solidToSTL(solid,longestMM=80){const k=longestMM/solid.size,p=solid.positions,ix=solid.indices,n=ix.length/3,buf=new ArrayBuffer(84+n*50),dv=new DataView(buf);dv.setUint32(80,n,true);
  let minz=1e9;for(let i=2;i<p.length;i+=3)if(p[i]<minz)minz=p[i];
  for(let t=0;t<n;t++){const v=[0,1,2].map(j=>{const i=ix[t*3+j];return[p[i*3]*k,p[i*3+1]*k,(p[i*3+2]-minz)*k]}),[a,b,c]=v,u=[b[0]-a[0],b[1]-a[1],b[2]-a[2]],w=[c[0]-a[0],c[1]-a[1],c[2]-a[2]];
    let N=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]];const l=Math.hypot(...N)||1;let o=84+t*50;
    for(const q of [N.map(x=>x/l),a,b,c])for(const x of q){dv.setFloat32(o,x,true);o+=4}}
  return buf}
