#!/usr/bin/env python3
"""Dead-simple attribute tagger for a COCO dataset (detect or segment).

Instances already exist in the COCO file; this tool only adds per-instance
**attributes**. Flow:

  1. Load the COCO file. If it has no attribute schema, the first screen lets you
     define attributes (name + comma-separated options).
  2. Then go image by image: click an object, pick its attribute values. Saved
     automatically, written back into each annotation's ``attributes`` field.
  3. Resume: re-run and it jumps to the first image that still has an untagged
     instance. (If attributes were never defined, you get the define screen.)

No third-party dependencies. Output defaults to ``<coco_stem>.tagged.json`` next
to the input and is loaded back on resume (input is never modified in place).

Usage:
    python tools/attr_tagger/tag.py --coco tools/attr_tagger/sample/coco.json \
        --images tools/attr_tagger/sample/images
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class Store:
    def __init__(self, coco, images_dir, out_path):
        self.coco = coco
        self.images_dir = Path(images_dir)
        self.out_path = Path(out_path)
        self.lock = threading.Lock()
        self._dirty = False

        coco.setdefault("annotations", [])
        coco.setdefault("categories", [])
        coco.setdefault("attributes", [])
        self.img_by_id = {im["id"]: im for im in coco.get("images", [])}
        self.cat_name = {c["id"]: c.get("name", str(c["id"])) for c in coco["categories"]}
        self.anns_by_img = {}
        for a in coco["annotations"]:
            if a.get("image_id") not in self.img_by_id:
                continue
            if not isinstance(a.get("attributes"), dict):
                a["attributes"] = {}
            self.anns_by_img.setdefault(a["image_id"], []).append(a)
        self.ann_by_id = {a["id"]: a for a in coco["annotations"] if "id" in a}
        self.image_ids = sorted(self.img_by_id)

        threading.Thread(target=self._autosave, daemon=True).start()

    # ---- schema ----
    def attr_names(self):
        return [d["name"] for d in self.coco["attributes"]]

    def attributes_payload(self):
        out = []
        for d in self.coco["attributes"]:
            opts = sorted(d.get("categories", []), key=lambda c: c["id"])
            out.append({"name": d["name"],
                        "options": [{"id": c["id"], "name": c.get("name", str(c["id"]))} for c in opts]})
        return out

    def define_attributes(self, attrs):
        """attrs = [{name, options:[str,...]}, ...] -> 0-based ids per option."""
        with self.lock:
            defs = []
            for a in attrs:
                name = a["name"].strip()
                opts = [o.strip() for o in a.get("options", []) if o.strip()]
                if not name or not opts:
                    continue
                defs.append({"name": name,
                             "categories": [{"id": i, "name": o} for i, o in enumerate(opts)]})
            self.coco["attributes"] = defs
            self._dirty = True

    # ---- tagging state ----
    def _untagged(self, a):
        """True if this instance is missing a value for any defined attribute."""
        return any(a["attributes"].get(n) is None for n in self.attr_names())

    def first_untagged_pos(self):
        for pos, iid in enumerate(self.image_ids):
            if any(self._untagged(a) for a in self.anns_by_img.get(iid, [])):
                return pos
        return 0

    def state(self):
        total = sum(len(v) for v in self.anns_by_img.values())
        tagged = sum(1 for v in self.anns_by_img.values() for a in v if not self._untagged(a))
        return {
            "has_attrs": bool(self.coco["attributes"]),
            "attributes": self.attributes_payload(),
            "num_images": len(self.image_ids),
            "start_pos": self.first_untagged_pos(),
            "total_instances": total, "tagged_instances": tagged,
        }

    def image_at(self, pos):
        if not self.image_ids:
            return None
        pos = max(0, min(pos, len(self.image_ids) - 1))
        iid = self.image_ids[pos]
        im = self.img_by_id[iid]
        insts = []
        for a in self.anns_by_img.get(iid, []):
            seg = a.get("segmentation")
            polys = seg if (isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple))) else None
            insts.append({
                "ann_id": a["id"], "bbox": a.get("bbox"), "polygons": polys,
                "category": self.cat_name.get(a.get("category_id"), str(a.get("category_id"))),
                "attributes": {n: a["attributes"].get(n) for n in self.attr_names()},
                "tagged": not self._untagged(a),
            })
        return {"pos": pos, "count": len(self.image_ids), "image_id": iid,
                "file_name": im["file_name"], "img_w": im.get("width"), "img_h": im.get("height"),
                "instances": insts}

    def next_untagged_pos(self, after):
        for pos in range(after + 1, len(self.image_ids)):
            iid = self.image_ids[pos]
            if any(self._untagged(a) for a in self.anns_by_img.get(iid, [])):
                return pos
        return None

    def set_attr(self, ann_id, attr, value):
        with self.lock:
            a = self.ann_by_id[ann_id]
            if value is None:
                a["attributes"].pop(attr, None)
            else:
                a["attributes"][attr] = value
            self._dirty = True

    def resolve_image(self, image_id):
        im = self.img_by_id.get(image_id)
        if not im:
            return None
        fn = im["file_name"]
        for cand in (self.images_dir / fn, self.images_dir / Path(fn).name):
            if cand.is_file():
                return cand
        return None

    # ---- persistence ----
    def save(self):
        with self.lock:
            tmp = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.coco, f, ensure_ascii=False)
            os.replace(tmp, self.out_path)
            self._dirty = False

    def _autosave(self):
        while True:
            time.sleep(10)
            if self._dirty:
                try:
                    self.save()
                except Exception as e:  # pragma: no cover
                    print("autosave failed:", e)


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Attribute Tagger</title>
<style>
 *{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:#111;color:#eee}
 #bar{display:flex;gap:10px;align-items:center;padding:8px 12px;background:#1b1b1b;border-bottom:1px solid #333;flex-wrap:wrap}
 button,input{font:inherit}
 button{background:#262626;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 10px;cursor:pointer}
 button:hover{background:#333}
 #wrap{display:flex;height:calc(100vh - 49px)}
 #cwrap{flex:1;display:flex;align-items:center;justify-content:center;background:#000;overflow:hidden}
 canvas{max-width:100%;max-height:100%}
 #side{width:320px;padding:14px;border-left:1px solid #333;overflow:auto}
 .grp{margin-bottom:16px}.grp h3{margin:0 0 6px;font-size:13px;color:#9bd}
 .opt{display:block;width:100%;text-align:left;margin:3px 0;padding:7px 10px;background:#222;border:1px solid #3a3a3a;border-radius:6px;color:#eee;cursor:pointer}
 .opt:hover{background:#2c2c2c}.opt.sel{background:#1d4ed8;border-color:#3b82f6}
 .opt .k{display:inline-block;width:16px;color:#888}
 #define{max-width:620px;margin:40px auto;padding:24px}
 #define h1{font-size:20px}.arow{display:flex;gap:8px;margin:8px 0}
 .arow input{flex:1;background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:8px}
 #status{color:#888;font-size:12px}#prog{color:#9bd;font-size:12px}
 .muted{color:#777;font-size:12px}kbd{background:#333;border:1px solid #555;border-radius:4px;padding:0 5px;font:11px monospace}
</style></head><body>
<div id="app"></div>
<script>
let S=null,CUR=null,POS=0,IMG=new Image(),SC=1,SEL=null;
const app=document.getElementById('app');

async function boot(){S=await (await fetch('/api/state')).json();S.has_attrs?startTag():defineScreen();}

// ---------- define attributes ----------
function defineScreen(){
 app.innerHTML=`<div id="define">
  <h1>Define attributes</h1>
  <p class="muted">No attributes in this dataset yet. Add one or more, each with comma-separated options. You only do this once.</p>
  <div id="rows"></div>
  <button onclick="addRow()">+ add attribute</button>
  <hr style="border-color:#333;margin:18px 0">
  <button style="background:#1d4ed8;border-color:#3b82f6" onclick="saveDefs()">Start tagging →</button>
  <span id="status"></span></div>`;
 addRow('','');addRow('','');
}
function addRow(n='',o=''){const r=document.createElement('div');r.className='arow';
 r.innerHTML=`<input placeholder="attribute name (e.g. color)" value="${n}"><input placeholder="options, comma-sep (e.g. red,green,blue)" value="${o}">`;
 document.getElementById('rows').appendChild(r);}
async function saveDefs(){
 const attrs=[...document.querySelectorAll('#rows .arow')].map(r=>{const i=r.querySelectorAll('input');
   return {name:i[0].value.trim(),options:i[1].value.split(',').map(s=>s.trim()).filter(Boolean)};}).filter(a=>a.name&&a.options.length);
 if(!attrs.length){document.getElementById('status').textContent='add at least one attribute with options';return;}
 await fetch('/api/define',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({attributes:attrs})});
 S=await (await fetch('/api/state')).json();startTag();
}

// ---------- tagging ----------
function startTag(){
 app.innerHTML=`<div id="bar">
   <button onclick="go(-1)">← Prev</button><span id="imgpos"></span><button onclick="go(1)">Next →</button>
   <button onclick="nextUntagged()">Next untagged image (n)</button>
   <span id="fname" class="muted"></span><span style="flex:1"></span>
   <span id="prog"></span><button onclick="saveNow()">Save (s)</button><span id="status"></span>
 </div>
 <div id="wrap"><div id="cwrap"><canvas id="cv"></canvas></div>
 <div id="side"><div id="sel" class="muted">Click an object to tag it.</div></div></div>`;
 POS=S.start_pos||0;loadImg();
}
async function loadImg(){
 const r=await fetch('/api/image?pos='+POS);if(r.status!==200)return;
 CUR=await r.json();POS=CUR.pos;SEL=null;
 document.getElementById('imgpos').textContent=`${POS+1}/${CUR.count}`;
 document.getElementById('fname').textContent=CUR.file_name;
 await new Promise(res=>{IMG=new Image();IMG.onload=res;IMG.onerror=res;IMG.src='/api/img?id='+CUR.image_id;});
 draw();renderSel();updateProg();
}
function updateProg(){const u=CUR.instances.filter(i=>!i.tagged).length;
 document.getElementById('prog').textContent=`this image: ${CUR.instances.length-u}/${CUR.instances.length} tagged`;}
const cv=()=>document.getElementById('cv');
function draw(){const c=cv();if(!c||!IMG.width)return;const ctx=c.getContext('2d');
 const W=c.parentElement.clientWidth-4,H=c.parentElement.clientHeight-4;
 SC=Math.min(W/IMG.width,H/IMG.height);c.width=IMG.width*SC;c.height=IMG.height*SC;
 ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(IMG,0,0,c.width,c.height);const T=v=>v*SC;
 CUR.instances.forEach(ins=>{const seld=SEL===ins.ann_id;
   const col=seld?'#ffd400':(ins.tagged?'#22c55e':'#4ea1ff');
   ctx.lineWidth=seld?3:1.8;ctx.strokeStyle=col;ctx.fillStyle=col+(seld?'44':'1f');
   if(ins.polygons){for(const p of ins.polygons){ctx.beginPath();for(let k=0;k<p.length;k+=2){const X=T(p[k]),Y=T(p[k+1]);k?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}ctx.closePath();ctx.fill();ctx.stroke();}}
   else if(ins.bbox){const[x,y,w,h]=ins.bbox;ctx.strokeRect(T(x),T(y),T(w),T(h));ctx.fillRect(T(x),T(y),T(w),T(h));}
   if(ins.bbox){const[x,y]=ins.bbox;ctx.font='12px system-ui';const t=ins.category+(ins.tagged?' ✓':'');const tw=ctx.measureText(t).width;
     ctx.fillStyle=col;ctx.fillRect(T(x),T(y)-15,tw+8,15);ctx.fillStyle='#000';ctx.fillText(t,T(x)+4,T(y)-4);}
 });
}
function ptInPoly(x,y,p){let c=false;for(let i=0,j=p.length-2;i<p.length;j=i,i+=2){const xi=p[i],yi=p[i+1],xj=p[j],yj=p[j+1];if(((yi>y)!=(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi)+xi))c=!c;}return c;}
function hit(ix,iy){const a=CUR.instances;for(let i=a.length-1;i>=0;i--){const o=a[i];
  if(o.polygons){if(o.polygons.some(p=>ptInPoly(ix,iy,p)))return o.ann_id;}
  else if(o.bbox){const[x,y,w,h]=o.bbox;if(ix>=x&&ix<=x+w&&iy>=y&&iy<=y+h)return o.ann_id;}}return null;}
document.addEventListener('click',e=>{const c=cv();if(!c||e.target!==c)return;
  const rc=c.getBoundingClientRect();const ix=(e.clientX-rc.left)/SC,iy=(e.clientY-rc.top)/SC;
  SEL=hit(ix,iy);draw();renderSel();});
function curInst(){return CUR.instances.find(i=>i.ann_id===SEL);}
function renderSel(){const host=document.getElementById('sel');const ins=curInst();
 if(!ins){host.innerHTML='<span class="muted">Click an object to tag it.</span>';return;}
 let h=`<div class="grp"><h3>${ins.category} <span class="muted">#${ins.ann_id}</span></h3></div>`;
 S.attributes.forEach((at)=>{const cur=ins.attributes[at.name];
   h+=`<div class="grp"><h3>${at.name}</h3>`;
   at.options.forEach((o,oi)=>{h+=`<button class="opt${cur===o.id?' sel':''}" onclick="setAttr('${at.name}',${o.id})"><span class="k">${oi<9?oi+1:''}</span>${o.name}</button>`;});
   h+=`<button class="opt${cur==null?' sel':''}" onclick="setAttr('${at.name}',null)"><span class="k">\`</span><i>clear</i></button></div>`;
 });
 h+=`<p class="muted">Tip: <kbd>1</kbd>..<kbd>9</kbd> set first attribute · click another object to continue.</p>`;
 host.innerHTML=h;
}
async function setAttr(name,val){const ins=curInst();if(!ins)return;
 await fetch('/api/set',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({ann_id:ins.ann_id,attr:name,value:val})});
 ins.attributes[name]=val;ins.tagged=S.attributes.every(a=>ins.attributes[a.name]!=null);
 draw();renderSel();updateProg();flash('saved');
}
async function go(d){POS+=d;if(POS<0)POS=0;await loadImg();}
async function nextUntagged(){const r=await (await fetch('/api/next?pos='+POS)).json();
 if(r.pos==null){flash('all images tagged 🎉');return;}POS=r.pos;await loadImg();}
async function saveNow(){await fetch('/api/save',{method:'POST'});flash('saved');}
function flash(t){const s=document.getElementById('status');if(s){s.textContent=t;clearTimeout(s._t);s._t=setTimeout(()=>s.textContent='',1000);}}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(!S||!S.has_attrs)return;
 if(e.key==='ArrowRight'){e.preventDefault();go(1);}else if(e.key==='ArrowLeft'){e.preventDefault();go(-1);}
 else if(e.key==='n'){e.preventDefault();nextUntagged();}else if(e.key==='s'){e.preventDefault();saveNow();}
 else if(e.key==='Escape'){SEL=null;draw();renderSel();}
 else if(/^[1-9]$/.test(e.key)){const ins=curInst();if(ins&&S.attributes[0]){const o=S.attributes[0].options[+e.key-1];if(o)setAttr(S.attributes[0].name,o.id);}}
 else if(e.key==='`'){const ins=curInst();if(ins&&S.attributes[0])setAttr(S.attributes[0].name,null);}
});
window.addEventListener('resize',draw);
boot();
</script></body></html>"""


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get("content-length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                b = HTML.encode()
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/api/state":
                self._json(store.state())
            elif u.path == "/api/image":
                d = store.image_at(int(q.get("pos", [0])[0]))
                self._json(d) if d else self._json({"error": "empty"}, 204)
            elif u.path == "/api/next":
                self._json({"pos": store.next_untagged_pos(int(q.get("pos", [0])[0]))})
            elif u.path == "/api/img":
                self._serve_image(int(q.get("id", [-1])[0]))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/api/define":
                store.define_attributes(self._body().get("attributes", []))
                self._json({"ok": True})
            elif u.path == "/api/set":
                d = self._body()
                store.set_attr(int(d["ann_id"]), d["attr"], d.get("value"))
                self._json({"ok": True})
            elif u.path == "/api/save":
                store.save()
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)

        def _serve_image(self, image_id):
            path = store.resolve_image(image_id)
            if not path:
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("content-type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            self.send_header("content-length", str(len(data)))
            self.send_header("cache-control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default=None, help="output JSON (default <coco_stem>.tagged.json; resumed if present)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    coco_path = Path(args.coco)
    out_path = Path(args.out) if args.out else coco_path.with_name(coco_path.stem + ".tagged.json")
    load_from = out_path if (out_path.exists() and out_path != coco_path) else coco_path
    print(f"loading {load_from} ...")
    coco = json.loads(Path(load_from).read_text())

    store = Store(coco, args.images, out_path)
    st = store.state()
    print(f"  {len(store.image_ids)} images · {st['total_instances']} instances · "
          f"attributes {'defined: ' + str(store.attr_names()) if st['has_attrs'] else 'NOT defined (define screen on open)'}")
    print(f"  output -> {out_path}")
    url = f"http://{args.host}:{args.port}"
    print(f"\n  open {url}\n  Ctrl-C to stop (saves on exit)\n")

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        store.save()
        print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
