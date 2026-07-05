"use strict";
const $ = s => document.querySelector(s);
const api = () => window.pywebview.api;

let flags = [];
let edits = new Map();        // sid -> text (mirror of backend for UI)
let segOriginal = new Map();  // sid -> original target

// ---------- model ----------
let modelChoice = "bert";
let backendChoice = "simalign";
$("#modelpick").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#modelpick").querySelectorAll("button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); modelChoice=b.dataset.v;
}));
$("#backendpick").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#backendpick").querySelectorAll("button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); backendChoice=b.dataset.v;
}));
$("#loadmodel").addEventListener("click", ()=>api().load_model(backendChoice, modelChoice));

window.addEventListener("model-status", e=>{
  const d=e.detail, dot=$("#mdot"), txt=$("#mtext");
  dot.className="status-dot "+(d.state==="ready"?"ready":d.state==="loading"?"loading":d.state==="error"?"error":"");
  if(d.state==="loading") txt.textContent=`Loading ${d.model} model (first run downloads it)…`;
  else if(d.state==="ready"){ txt.textContent=`Model ready: ${d.model}.`; maybeEnableRun(); }
  else if(d.state==="error"){ txt.textContent="Model error: "+d.error; }
});

// ---------- files ----------
let fileList=[];
$("#openfiles").addEventListener("click", async ()=>{
  const res=await api().open_files();
  fileList=res.files||[]; renderChips(); maybeEnableRun();
});
function renderChips(){
  const box=$("#filechips"); box.innerHTML="";
  let segs=0;
  fileList.forEach(f=>{
    segs+=f.segments||0;
    const c=document.createElement("span");
    c.className="chip";
    c.innerHTML=`${esc(f.name)} <span style="color:var(--mut)">· ${f.segments}</span> <span class="x" data-n="${esc(f.name)}">✕</span>`;
    box.appendChild(c);
  });
  $("#filehint").textContent = fileList.length ? `${fileList.length} file(s), ${segs} segments loaded.` : "Select one or more .xlf / .xliff files.";
  box.querySelectorAll(".x").forEach(x=>x.addEventListener("click",async ev=>{
    const r=await api().remove_file(ev.target.dataset.n); fileList=r.files||[]; renderChips(); maybeEnableRun();
  }));
}
function maybeEnableRun(){
  const ready=$("#mdot").classList.contains("ready");
  $("#run").disabled = !(ready && fileList.length);
}

// ---------- n-gram controls ----------
let nMode="range", nLow=2, nHigh=3;
const nchips=$("#nchips");
function paintChips(){
  nchips.querySelectorAll("button").forEach(b=>{
    const n=+b.dataset.n; b.classList.remove("on","edge");
    if(nMode==="exact"){ if(n===nLow)b.classList.add("on"); }
    else{ if(n===nLow||n===nHigh)b.classList.add("edge"); else if(n>nLow&&n<nHigh)b.classList.add("on"); }
  });
  $("#nlabel").textContent = nMode==="exact" ? `exactly ${nLow} word${nLow>1?"s":""}` : `${nLow}–${nHigh} words`;
}
nchips.querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  const n=+b.dataset.n;
  if(nMode==="exact"){ nLow=nHigh=n; }
  else { if(n<=nLow)nLow=n; else if(n>=nHigh)nHigh=n; else (n-nLow<=nHigh-n)?nLow=n:nHigh=n; }
  paintChips();
}));
$("#nmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#nmode").querySelectorAll("button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); nMode=b.dataset.v; if(nMode==="exact")nHigh=nLow; paintChips();
}));
let swMode="trim";
$("#swmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#swmode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); swMode=b.dataset.v;
}));
let foldTaa=true;
$("#taamode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#taamode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); foldTaa=b.dataset.v==="on";
}));
let stripClitics=true;
$("#clitmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#clitmode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); stripClitics=b.dataset.v==="on";
}));
let clusterOn=true;
$("#clustermode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#clustermode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); clusterOn=b.dataset.v==="on";
}));
let reverseOn=false;
$("#revmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#revmode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); reverseOn=b.dataset.v==="on";
}));
$("#minvar").addEventListener("input",e=>$("#minvarlabel").textContent=e.target.value+"×");
$("#minocc").addEventListener("input",e=>$("#occlabel").textContent=e.target.value+"×");

// ---------- analyze ----------
$("#run").addEventListener("click", ()=>{
  $("#progress").classList.remove("hidden"); $("#progbar").style.width="0%";
  $("#run").disabled=true;
  api().analyze({nmin:nLow,nmax:nHigh,stop_mode:swMode,min_occurrences:+$("#minocc").value,fold_taa:foldTaa,strip_clitics:stripClitics,cluster_spans:clusterOn,min_variant_count:+$("#minvar").value,reverse:reverseOn});
});
window.addEventListener("analyze-progress", e=>{
  const {done,total}=e.detail; $("#progbar").style.width=(100*done/total).toFixed(1)+"%";
});
window.addEventListener("analyze-error", e=>{
  $("#progress").classList.add("hidden"); $("#run").disabled=false;
  alert("Analysis error: "+e.detail.error);
});
let reverseFlags=[];
window.addEventListener("analyze-done", e=>{
  $("#progress").classList.add("hidden"); $("#run").disabled=false;
  const d=e.detail; flags=d.flags; reverseFlags=d.reverse||[];
  // capture originals & current edits
  segOriginal.clear();
  flags.forEach(f=>f.variants.forEach(v=>v.occurrences.forEach(o=>{
    segOriginal.set(o.sid,o.original);
    if(o.target!==o.original) edits.set(o.sid,o.target);
  })));
  $("#summary").classList.remove("hidden");
  $("#s-seg").textContent=d.segments; $("#s-files").textContent=d.files; $("#s-flag").textContent=flags.length;
  $("#s-rev").textContent=reverseFlags.length; $("#s-ph").textContent=d.placeholder_issues||0;
  $("#toolsrow").classList.remove("hidden");
  $("#llmall").classList.toggle("hidden", !$("#llm-status").dataset.ok);
  $("#auxout").innerHTML="";
  viewMode="fwd";
  $("#viewmode").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x.dataset.v==="fwd"));
  render(); syncToggle(); refreshDirty();
});

// ---------- render ----------
const filterInput=$("#filter");
filterInput.addEventListener("input",()=>{render();syncToggle();});
function render(){
  const host=$("#results");
  $("#filterbar").classList.toggle("hidden", flags.length===0 && !filterInput.value);
  const q=filterInput.value.trim().toLowerCase();
  const data = q ? flags.filter(f=>f.ngram.toLowerCase().includes(q)||f.variants.some(v=>v.span.includes(q))) : flags;
  if(!flags.length){ host.innerHTML=`<div class="empty"><div class="big">✓</div><strong>No inconsistencies found</strong><div style="margin-top:6px">Every aligned span is consistent under these settings.</div></div>`; $("#showing").textContent=""; return; }
  $("#showing").textContent=`${data.length} of ${flags.length} shown`;
  host.innerHTML=data.map((f,gi)=>{
    const vars=f.variants.map((v,vi)=>{
      const rows=v.occurrences.map(o=>{
        const cur=edits.has(o.sid)?edits.get(o.sid):o.target;
        const ch=edits.has(o.sid)&&edits.get(o.sid)!==segOriginal.get(o.sid);
        return `<div class="seg ${ch?'changed':''}" data-sid="${esc(o.sid)}">
          <div class="seg-src">${hl(o.source,q)}</div>
          <textarea class="seg-tgt" data-sid="${esc(o.sid)}" rows="1">${esc(cur)}</textarea>
          <div class="seg-meta">
            <button class="revert-seg" data-sid="${esc(o.sid)}">↺</button>
            <span>${esc(o.file)}·${esc(o.unit)}</span>
          </div></div>`;
      }).join("");
      return `<div class="vargroup">
        <div class="varhead">
          <span class="vartag">Variant ${vi+1}</span>
          <span class="varspan">${esc(v.span)}</span>
          <span class="cnt">${v.count}×</span>
          <button class="usebtn" data-g="${gi}" data-v="${vi}">Use for all ${f.total}</button>
        </div>
        <div class="segs">${rows}</div></div>`;
    }).join("");
    return `<div class="group">
      <div class="group-head" data-g="${gi}">
        <span class="chev">▸</span><span class="badge">${f.distinct} spans</span>
        <span class="src">${hl(f.ngram,q)}</span>
        <span class="meta">${f.total} occ · ${Math.round((f.score||0)*100)}% split</span>
        <button class="llmbtn ${$("#llm-status").dataset.ok?'':'hidden'}" data-g="${gi}">LLM check</button>
      </div>
      <div class="variants hidden">${vars}</div>
      <div class="llmout hidden" data-g="${gi}"></div>
    </div>`;
  }).join("");
  bind(host,data);
}
function bind(host,data){
  host.querySelectorAll(".group-head").forEach(h=>h.addEventListener("click",ev=>{
    if(ev.target.classList.contains("llmbtn")) return;
    const panel=h.nextElementSibling, open=panel.classList.toggle("hidden");
    h.querySelector(".chev").textContent=open?"▸":"▾";
    if(!open) panel.querySelectorAll("textarea.seg-tgt").forEach(grow);
  }));
  host.querySelectorAll("textarea.seg-tgt").forEach(ta=>ta.addEventListener("input",()=>{
    const sid=ta.dataset.sid;
    if(ta.value===segOriginal.get(sid)) edits.delete(sid); else edits.set(sid,ta.value);
    ta.closest(".seg").classList.toggle("changed",edits.has(sid));
    api().set_edit(sid,ta.value); grow(ta); refreshDirty();
  }));
  host.querySelectorAll(".revert-seg").forEach(b=>b.addEventListener("click",e=>{
    e.stopPropagation(); const sid=b.dataset.sid, orig=segOriginal.get(sid);
    edits.delete(sid); api().revert([sid]);
    const ta=b.closest(".seg").querySelector("textarea"); ta.value=orig;
    b.closest(".seg").classList.remove("changed"); grow(ta); refreshDirty();
  }));
  host.querySelectorAll(".usebtn").forEach(btn=>btn.addEventListener("click",e=>{
    e.stopPropagation(); const f=data[+btn.dataset.g], v=f.variants[+btn.dataset.v], text=v.span;
    const group=btn.closest(".group");
    f.variants.forEach(vv=>vv.occurrences.forEach(o=>{
      if(text===segOriginal.get(o.sid)) edits.delete(o.sid); else edits.set(o.sid,text);
      api().set_edit(o.sid,text);
      const ta=group.querySelector(`textarea.seg-tgt[data-sid="${cssEsc(o.sid)}"]`);
      if(ta){ta.value=text; ta.closest(".seg").classList.toggle("changed",edits.has(o.sid)); grow(ta);}
    }));
    refreshDirty();
  }));
  host.querySelectorAll(".llmbtn").forEach(btn=>btn.addEventListener("click",async e=>{
    e.stopPropagation(); const f=data[+btn.dataset.g];
    const out=btn.closest(".group").querySelector(".llmout");
    out.classList.remove("hidden","bad"); out.textContent="Asking the model…";
    const res=await api().llm_judge(f.ngram, f.variants.map(v=>v.span));
    if(res.error){ out.classList.add("bad"); out.textContent="LLM error: "+res.error; return; }
    if(res.verdict){ out.textContent=`Verdict: ${res.verdict}${res.preferred?` · preferred: ${res.preferred}`:''}${res.reason?` — ${res.reason}`:''}`; }
    else out.textContent="Model response: "+(res.raw||JSON.stringify(res));
  }));
}
function grow(ta){ta.style.height="auto";ta.style.height=ta.scrollHeight+"px";}

// ---------- expand/collapse all ----------
const toggleAll=$("#toggleall");
toggleAll.addEventListener("click",()=>{
  const expand=toggleAll.dataset.state==="collapsed";
  document.querySelectorAll("#results .group").forEach(g=>{
    const p=g.querySelector(".variants"); p.classList.toggle("hidden",!expand);
    g.querySelector(".chev").textContent=expand?"▾":"▸";
    if(expand) p.querySelectorAll("textarea.seg-tgt").forEach(grow);
  });
  toggleAll.dataset.state=expand?"expanded":"collapsed";
  toggleAll.textContent=expand?"Collapse all":"Expand all";
});
function syncToggle(){toggleAll.dataset.state="collapsed";toggleAll.textContent="Expand all";}

// ---------- dirty / export ----------
function refreshDirty(){
  const n=edits.size, bar=$("#editbar");
  if(n>0){ bar.classList.remove("hidden");
    const files=new Set([...edits.keys()].map(sid=>sid.split("#")[0]));
    $("#editcount").textContent=`${n} segment(s) edited across ${files.size} file(s)`;
  } else bar.classList.add("hidden");
}
$("#revertall").addEventListener("click", async ()=>{
  if(!edits.size) return;
  if(!confirm(`Revert all ${edits.size} edited segment(s)?`)) return;
  await api().revert_all(); edits.clear();
  document.querySelectorAll("textarea.seg-tgt").forEach(ta=>{
    const o=segOriginal.get(ta.dataset.sid); if(o!==undefined){ta.value=o;ta.closest(".seg").classList.remove("changed");grow(ta);}
  });
  refreshDirty();
});
$("#exportbtn").addEventListener("click", async ()=>{
  const r=await api().export();
  if(r.ok) alert(`Exported ${r.written.length} file(s) to:\n${r.dir}\n\n${r.written.join("\n")}`);
  else alert(r.msg||"Export failed.");
});

// ---------- LLM config ----------
$("#llm-test").addEventListener("click", async ()=>{
  const url=$("#llm-url").value.trim(), key=$("#llm-key").value.trim(), model=$("#llm-model").value.trim();
  const provider=$("#llm-provider").value;
  $("#llm-status").textContent="Testing…";
  const r=await api().set_llm(url,key,model,provider);
  if(r.ok){ $("#llm-status").textContent="Connected ✓"; $("#llm-status").dataset.ok="1"; render(); }
  else { $("#llm-status").textContent="Failed: "+(r.error||r.msg||"check fields"); delete $("#llm-status").dataset.ok; }
});

// ---------- view switch: forward / reverse ----------
let viewMode="fwd";
$("#viewmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#viewmode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on");
  viewMode=b.dataset.v;
  if(viewMode==="rev") renderReverse(); else { render(); syncToggle(); }
}));
function renderReverse(){
  const host=$("#results"); $("#filterbar").classList.add("hidden");
  if(!reverseFlags.length){ host.innerHTML=`<div class="empty"><div class="big">✓</div><strong>No overloaded spans</strong><div style="margin-top:6px">No Arabic span translates more than one English term.<br>(Enable "+ AR→EN" before analyzing to compute this.)</div></div>`; return; }
  host.innerHTML=reverseFlags.map(f=>{
    const uses=f.uses.map(u=>`<div class="vargroup"><div class="varhead">
      <span class="vartag">Term</span><span class="varspan" dir="ltr">${esc(u.term)}</span><span class="cnt">${u.count}×</span></div></div>`).join("");
    return `<div class="group"><div class="group-head">
      <span class="badge">${f.distinct} terms</span>
      <span class="varspan">${esc(f.span)}</span>
      <span class="meta">${f.total} occ · ${Math.round((f.score||0)*100)}% split</span>
    </div><div class="variants">${uses}</div></div>`;
  }).join("");
}

// ---------- glossary ----------
$("#loadgloss").addEventListener("click", async ()=>{
  const r=await api().load_glossary();
  $("#toolshint").textContent = r.ok ? `Glossary: ${r.entries} term(s) loaded.` : (r.error||"No glossary loaded.");
});
$("#checkgloss").addEventListener("click", async ()=>{
  const r=await api().check_glossary();
  const out=$("#auxout");
  if(!r.ok){ out.innerHTML=`<div class="empty">${esc(r.msg||"Load a glossary first.")}</div>`; return; }
  if(!r.count){ out.innerHTML=`<div class="empty"><div class="big">✓</div><strong>Glossary adherence: no violations</strong></div>`; return; }
  out.innerHTML=`<div class="group"><div class="group-head"><span class="badge">${r.count}</span><span class="src">Glossary violations</span></div><div class="variants">`+
    r.violations.map(v=>`<div class="seg"><div class="seg-src">${hl(v.source,v.term.toLowerCase())} <span style="color:var(--mut)">— expected <b>${esc(v.approved)}</b></span></div><div dir="rtl" style="margin-top:4px">${esc(v.target)}</div><div class="seg-meta"><span>${esc(v.sid)}</span></div></div>`).join("")+
    `</div></div>`;
});

// ---------- placeholder report ----------
$("#ph-k").addEventListener("click", async ()=>{
  const r=await api().placeholder_report();
  const out=$("#auxout");
  if(!r.count){ out.innerHTML=`<div class="empty"><div class="big">✓</div><strong>No placeholder mismatches</strong></div>`; return; }
  out.innerHTML=`<div class="group"><div class="group-head"><span class="badge">${r.count}</span><span class="src">Placeholder mismatches (source vs target)</span></div><div class="variants">`+
    r.items.map(s=>`<div class="seg"><div class="seg-src">${esc(s.source)}</div><div dir="rtl" style="margin-top:4px">${esc(s.target)}</div><div class="seg-meta"><span>src: ${esc((s.src_ph||[]).join(", ")||"—")} · tgt: ${esc((s.tgt_ph||[]).join(", ")||"—")}</span></div></div>`).join("")+
    `</div></div>`;
});

// ---------- LLM: judge all ----------
$("#llmall").addEventListener("click", async ()=>{
  $("#llmall").disabled=true; $("#toolshint").textContent="Asking the model about every flag…";
  const r=await api().llm_judge_all();
  $("#llmall").disabled=false;
  if(r.error){ $("#toolshint").textContent="LLM error: "+r.error; return; }
  const byNgram=new Map(r.verdicts.map(v=>[v.ngram,v]));
  document.querySelectorAll("#results .group").forEach((g,i)=>{
    const f=flags[i]; if(!f) return; const v=byNgram.get(f.ngram); if(!v) return;
    let out=g.querySelector(".llmout"); if(out){ out.classList.remove("hidden","bad");
      out.textContent = v.error ? ("LLM error: "+v.error) : `Verdict: ${v.verdict||"?"}${v.preferred?` · preferred: ${v.preferred}`:''}${v.reason?` — ${v.reason}`:''}`;
      if(v.error||v.verdict==="inconsistent") out.classList.add("bad"); }
  });
  $("#toolshint").textContent=`LLM judged ${r.verdicts.length} flag(s).`;
});

// ---------- utils ----------
function esc(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function cssEsc(s){return (window.CSS&&CSS.escape)?CSS.escape(s):s.replace(/["\\#.:]/g,"\\$&");}
function hl(s,q){const e=esc(s);if(!q)return e;try{return e.replace(new RegExp("("+q.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")+")","ig"),"<mark>$1</mark>");}catch{return e;}}

paintChips();
