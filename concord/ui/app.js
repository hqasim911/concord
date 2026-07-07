"use strict";
const $ = s => document.querySelector(s);
const api = () => window.pywebview.api;

// ---------- page navigation ----------
function showPage(p){
  document.querySelectorAll(".page").forEach(el=>el.classList.add("hidden"));
  $("#page-"+p).classList.remove("hidden");
  document.querySelectorAll(".nav .step").forEach(b=>b.classList.toggle("on",b.dataset.p===p));
}
$("#nav-settings").addEventListener("click",()=>showPage("settings"));
$("#nav-run").addEventListener("click",()=>{ if(!$("#nav-run").disabled) showPage("run"); });
$("#nav-vault").addEventListener("click",()=>{ showPage("vault"); renderVault(); });
$("#back-settings").addEventListener("click",()=>showPage("settings"));

// ---------- under-the-hood console ----------
function clog(msg,cls){
  const c=$("#console"); if(!c) return;
  const now=new Date().toLocaleTimeString();
  const line=document.createElement("div");
  line.innerHTML=`<span class="ts">${now}</span>  <span class="${cls||''}">${esc(msg)}</span>`;
  c.appendChild(line); c.scrollTop=c.scrollHeight;
}

let flags = [];
let lastRendered = [];        // the slice currently in the DOM (post-filter/cap)
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
let ensembleMode="intersect";
$("#ensmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#ensmode").querySelectorAll("button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); ensembleMode=b.dataset.v;
}));
$("#loadmodel").addEventListener("click", ()=>api().load_model(backendChoice, modelChoice, ensembleMode));

window.addEventListener("model-status", e=>{
  const d=e.detail, dot=$("#mdot"), txt=$("#mtext");
  dot.className="status-dot "+(d.state==="ready"?"ready":d.state==="loading"?"loading":d.state==="error"?"error":"");
  if(d.state==="loading"){ txt.textContent=`Loading ${d.model} model (first run downloads it)…`; clog(`Loading ${d.model} model…`); }
  else if(d.state==="ready"){ txt.textContent=`Model ready: ${d.model}.`; clog(`Model ready: ${d.model}`,"em"); maybeEnableRun(); }
  else if(d.state==="error"){ txt.textContent="Model error: "+d.error; clog("Model error: "+d.error,"err"); }
});

// ---------- files ----------
let fileList=[];
$("#openfiles").addEventListener("click", async ()=>{
  const res=await api().open_files();
  fileList=res.files||[]; renderChips(); maybeEnableRun(); refreshViewer();
});
async function refreshViewer(){
  const box=$("#viewer");
  if(!fileList.length){ box.innerHTML=""; return; }
  const r=await api().list_segments(2000);
  if(!r.total){ box.innerHTML=""; $("#batchhint").textContent="of the file"; return; }
  const note=`${r.total} segment(s)${r.shown<r.total?` · showing first ${r.shown}`:''}`;
  box.innerHTML=`<div class="vhint">${esc(note)}</div>`+r.segments.map(s=>
    `<div class="vrow"><span class="vi">${esc(s.sid)}</span><span class="vsrc">${esc(s.source)}</span><span class="vtgt">${esc(s.target)}</span></div>`).join("");
  window.segTotal=r.total; updateBatchHint();
}
function updateBatchHint(){
  const size=+$("#batchsize").value, num=Math.max(+$("#batchnum").value||1,1), total=window.segTotal||0;
  const el=$("#batchhint");
  if(size>0){
    const start=(num-1)*size, end=total?Math.min(start+size,total):start+size;
    el.textContent = start>=(total||Infinity) && total ? `past end (only ${total} segments)` :
      `segments ${start+1}–${end}${total?` of ${total}`:''}`;
  } else { el.textContent = total?`whole file (${total} segments)`:"whole file"; }
}
$("#batchsize").addEventListener("input",updateBatchHint);
$("#batchnum").addEventListener("input",updateBatchHint);
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
    const r=await api().remove_file(ev.target.dataset.n); fileList=r.files||[]; renderChips(); maybeEnableRun(); refreshViewer();
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
let containOn=true;
$("#containmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#containmode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); containOn=b.dataset.v==="on";
}));
let reverseOn=false;
$("#revmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#revmode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); reverseOn=b.dataset.v==="on";
}));
let includeAll=true;
$("#incmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#incmode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); includeAll=b.dataset.v==="all";
}));
let prefilterOn=false;
$("#prefiltermode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#prefiltermode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); prefilterOn=b.dataset.v==="on";
}));
$("#prefilterthr").addEventListener("input",e=>$("#thrlabel").textContent=e.target.value+"%");
let faithOn=false;
$("#faithmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#faithmode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); faithOn=b.dataset.v==="on";
}));
$("#faiththr").addEventListener("input",e=>$("#faithlabel").textContent=e.target.value+"%");
let checkTB=false;
$("#tbmode").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#tbmode").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); checkTB=b.dataset.v==="on";
}));
function refreshTB(){ if(!window.pywebview) return; api().termbase_info().then(r=>{
  $("#tbcount").textContent=`${r.count} approved term(s)`;
  $("#vault-count").textContent=`${r.count} approved term(s) · ~/.concord/termbase.json`;
}); }
window.addEventListener("pywebviewready", refreshTB);
$("#minvar").addEventListener("input",e=>$("#minvarlabel").textContent=e.target.value+"×");
$("#minocc").addEventListener("input",e=>$("#occlabel").textContent=e.target.value+"×");

// ---------- analyze ----------
$("#run").addEventListener("click", ()=>{
  $("#nav-run").disabled=false; showPage("run");
  $("#console").innerHTML=""; clog("Analysis started","em");
  $("#summary").classList.add("hidden"); $("#toolsrow").classList.add("hidden");
  $("#filterbar").classList.add("hidden"); $("#results").innerHTML=""; $("#auxout").innerHTML="";
  $("#progress").classList.remove("hidden"); $("#progbar").style.width="0%"; $("#progpct").textContent="0%";
  $("#runphase").textContent="Running…";
  $("#run").disabled=true;
  api().analyze({nmin:nLow,nmax:nHigh,stop_mode:swMode,min_occurrences:+$("#minocc").value,fold_taa:foldTaa,strip_clitics:stripClitics,cluster_spans:clusterOn,merge_contained:containOn,min_variant_count:+$("#minvar").value,reverse:reverseOn,include_consistent:includeAll,labse_prefilter:prefilterOn,prefilter_threshold:(+$("#prefilterthr").value)/100,faithfulness_filter:faithOn,faithfulness_threshold:(+$("#faiththr").value)/100,check_termbase:checkTB,batch_size:+$("#batchsize").value,batch_num:+$("#batchnum").value});
});
window.addEventListener("analyze-log", e=>clog(e.detail.msg));
window.addEventListener("analyze-progress", e=>{
  const {done,total}=e.detail;
  const pct=total?100*done/total:0;
  $("#progbar").style.width=pct.toFixed(1)+"%";
  $("#progpct").textContent=Math.round(pct)+"%";
  $("#runphase").textContent=`Aligning ${done}/${total} unique sentence pairs…`;
});
window.addEventListener("analyze-error", e=>{
  $("#progress").classList.add("hidden"); $("#run").disabled=false;
  $("#runphase").textContent="Analysis failed.";
  clog("Analysis error: "+e.detail.error,"err");
  if(e.detail.trace) clog(e.detail.trace,"err");
});
let reverseFlags=[];
window.addEventListener("analyze-done", e=>{
  $("#progbar").style.width="100%"; $("#progpct").textContent="100%";
  $("#run").disabled=false;
  const d=e.detail; flags=d.flags; reverseFlags=d.reverse||[];
  // capture originals & current edits
  segOriginal.clear();
  flags.forEach(f=>f.variants.forEach(v=>v.occurrences.forEach(o=>{
    segOriginal.set(o.sid,o.original);
    if(o.target!==o.original) edits.set(o.sid,o.target);
  })));
  $("#summary").classList.remove("hidden");
  $("#s-seg").textContent=d.segments; $("#s-files").textContent=d.files;
  $("#s-ngrams").textContent=flags.length; $("#s-flag").textContent=d.inconsistent||0;
  $("#s-rev").textContent=reverseFlags.length; $("#s-ph").textContent=d.placeholder_issues||0;
  $("#s-tbv").textContent=flags.filter(f=>f.tb_violation).length;
  $("#runphase").textContent=`Done — ${flags.length} n-gram(s), ${d.inconsistent||0} inconsistent.`;
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
$("#rx").addEventListener("change",()=>{render();syncToggle();});
$("#inconly").addEventListener("change",()=>{render();syncToggle();});
function buildMatcher(){
  const raw=filterInput.value.trim();
  if(!raw){ filterInput.classList.remove("rxbad"); return null; }
  if($("#rx").checked){
    try{ const re=new RegExp(raw,"i"); filterInput.classList.remove("rxbad"); return s=>re.test(String(s)); }
    catch(err){ filterInput.classList.add("rxbad"); return null; }
  }
  filterInput.classList.remove("rxbad");
  const low=raw.toLowerCase(); return s=>String(s).toLowerCase().includes(low);
}
const RENDER_CAP=400;
function render(){
  const host=$("#results");
  if(!flags.length){ $("#filterbar").classList.add("hidden"); host.innerHTML=`<div class="empty"><div class="big">✓</div><strong>No n-grams found</strong><div style="margin-top:6px">Nothing matched these settings.</div></div>`; $("#showing").textContent=""; return; }
  $("#filterbar").classList.remove("hidden");
  const m=buildMatcher();
  let data=flags;
  if($("#inconly").checked) data=data.filter(f=>f.inconsistent);
  if(m) data=data.filter(f=>m(f.ngram)||f.variants.some(v=>m(v.span)));
  const shown=data.slice(0,RENDER_CAP);
  lastRendered=shown;
  const q=$("#rx").checked?"":filterInput.value.trim();
  $("#showing").textContent = data.length>RENDER_CAP ? `showing ${shown.length} of ${data.length} — refine search` : `${shown.length} of ${flags.length} shown`;
  if(!shown.length){ host.innerHTML=`<div class="empty">No matches.</div>`; return; }
  host.innerHTML=shown.map((f,gi)=>{
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
          <button class="approvebtn ${f.approved===v.span?'done':''}" data-g="${gi}" data-v="${vi}">${f.approved===v.span?'Approved ✓':'Approve ✓'}</button>
        </div>
        <div class="segs">${rows}</div></div>`;
    }).join("");
    return `<div class="group ${f.inconsistent?'':'consistent'} ${f.tb_violation?'violation':''}">
      <div class="group-head" data-g="${gi}">
        <span class="chev">▸</span><span class="badge">${f.tb_violation?'vault':f.distinct+' span'+(f.distinct>1?'s':'')}</span>
        <span class="src">${hl(f.ngram,q)}</span>
        <span class="meta">${f.total} occ${f.inconsistent?` · ${Math.round((f.score||0)*100)}% split`:' · consistent'}${f.verify?` · LaBSE ${esc(f.verify.verdict)}${f.verify.agreement!=null?` (${f.verify.agreement})`:''}`:''}</span>
        ${f.inconsistent?`<button class="whybtn" data-g="${gi}">why flagged?</button>`:''}
        <button class="llmbtn ${$("#llm-status").dataset.ok?'':'hidden'}" data-g="${gi}">LLM check</button>
      </div>
      ${f.approved?`<div class="tbnote ${f.tb_violation?'bad':''}">${f.tb_violation?'⚠ Vault violation — ':'✓ Matches vault — '}approved: <span dir="rtl">${esc(f.approved)}</span>${f.tb_violation?` · this file uses ${f.variants.filter(v=>v.span!==f.approved).map(v=>`<span dir="rtl">${esc(v.span)}</span>`).join(", ")}`:''}</div>`:''}
      ${f.dropped&&f.dropped.length?`<div class="dropnote">Dropped ${f.dropped.length} mis-aligned span(s) — not a translation of the term: ${f.dropped.map(d=>`<span dir="rtl">${esc(d.span)}</span> (sim ${d.sim})`).join(", ")}</div>`:''}
      <div class="variants hidden">${vars}</div>
      <div class="whyout hidden" data-g="${gi}"></div>
      <div class="mtout hidden" data-g="${gi}"></div>
      <div class="llmout hidden" data-g="${gi}"></div>
    </div>`;
  }).join("");
  bind(host,shown);
}
function bind(host,data){
  host.querySelectorAll(".group-head").forEach(h=>h.addEventListener("click",ev=>{
    if(ev.target.classList.contains("llmbtn")||ev.target.classList.contains("whybtn")) return;
    const panel=h.nextElementSibling, open=panel.classList.toggle("hidden");
    h.querySelector(".chev").textContent=open?"▸":"▾";
    if(!open) panel.querySelectorAll("textarea.seg-tgt").forEach(grow);
  }));
  host.querySelectorAll(".whybtn").forEach(btn=>btn.addEventListener("click",e=>{
    e.stopPropagation();
    const f=data[+btn.dataset.g];
    const out=btn.closest(".group").querySelector(".whyout");
    if(!out.classList.contains("hidden")){ out.classList.add("hidden"); return; }
    out.innerHTML=whyFlagged(f); out.classList.remove("hidden");
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
  host.querySelectorAll(".approvebtn").forEach(btn=>btn.addEventListener("click",e=>{
    e.stopPropagation();
    const f=data[+btn.dataset.g], v=f.variants[+btn.dataset.v];
    api().approve_term(f.ngram, v.span).then(()=>refreshTB());
    f.approved=v.span;
    btn.closest(".variants").querySelectorAll(".approvebtn").forEach(b=>{b.classList.remove("done");b.textContent="Approve ✓";});
    btn.classList.add("done"); btn.textContent="Approved ✓";
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

// ---------- approved term base ----------
$("#approveall").addEventListener("click", async ()=>{
  const inc=flags.filter(f=>f.inconsistent).length;
  if(!inc){ $("#toolshint").textContent="No inconsistent flags to approve."; return; }
  if(!confirm(`Approve the most-frequent translation of ${inc} inconsistent flag(s) into the term base? You can prune it afterwards.`)) return;
  const r=await api().approve_all(); refreshTB();
  flags.forEach(f=>{ if(f.inconsistent) f.approved=f.variants[0].span; });
  render();
  $("#toolshint").textContent=`Approved ${r.approved} term(s) · ${r.count} in term base.`;
});
// Settings "Open vault →" button jumps to the vault page
$("#tbview").addEventListener("click", ()=>{ showPage("vault"); renderVault(); });

// ---------- N-gram Vault page ----------
let vaultEntries=[];
async function renderVault(){
  const r=await api().termbase_info();
  vaultEntries=r.entries||[];
  refreshTB();
  paintVault();
}
function vaultMatcher(){
  const raw=$("#vault-search").value.trim(), el=$("#vault-search");
  if(!raw){ el.classList.remove("rxbad"); return null; }
  if($("#vault-rx").checked){
    try{ const re=new RegExp(raw,"i"); el.classList.remove("rxbad"); return s=>re.test(String(s)); }
    catch(e){ el.classList.add("rxbad"); return null; }
  }
  el.classList.remove("rxbad");
  const low=raw.toLowerCase(); return s=>String(s).toLowerCase().includes(low);
}
function jsKey(s){ return s.toLowerCase().replace(/\s+/g," ").trim(); }
function paintVault(){
  const box=$("#vault-list"), m=vaultMatcher();
  if(!vaultEntries.length){ $("#vault-shown").textContent=""; box.innerHTML=`<div class="empty"><div class="big">🗄</div><strong>Vault is empty</strong><div style="margin-top:6px">Approve terms from the results page, or add them above.</div></div>`; return; }
  let data=vaultEntries;
  if(m) data=data.filter(e=>m(e.source)||m(e.target));
  $("#vault-shown").textContent=`${data.length} of ${vaultEntries.length}`;
  if(!data.length){ box.innerHTML=`<div class="empty">No matches.</div>`; return; }
  box.innerHTML=data.map(e=>`<div class="vrowe" data-k="${esc(e.key)}">
    <input class="vsrc-i" value="${esc(e.source)}">
    <input class="vtgt-i" value="${esc(e.target)}" dir="rtl">
    <span class="vmeta">${esc((e.updated||'').slice(0,10))}</span>
    <button class="va del">✕</button>
  </div>`).join("");
  box.querySelectorAll(".vrowe").forEach(row=>{
    let key=row.dataset.k;
    const e=vaultEntries.find(x=>x.key===key);
    const si=row.querySelector(".vsrc-i"), ti=row.querySelector(".vtgt-i");
    async function save(){
      const s=si.value.trim(), t=ti.value.trim();
      if(!s||!t) return;
      if(e && s===e.source && t===e.target) return;    // unchanged
      await api().update_term(key, s, t);
      if(e){ e.source=s; e.target=t; e.key=jsKey(s); }
      key=jsKey(s); row.dataset.k=key;
      row.classList.add("saved"); setTimeout(()=>row.classList.remove("saved"),700);
      refreshTB();
    }
    si.addEventListener("blur",save); ti.addEventListener("blur",save);
    si.addEventListener("keydown",ev=>{ if(ev.key==="Enter") si.blur(); });
    ti.addEventListener("keydown",ev=>{ if(ev.key==="Enter") ti.blur(); });
    row.querySelector(".del").addEventListener("click",async()=>{
      if(!confirm("Remove this entry?")) return;
      await api().remove_term(key); await renderVault();
    });
  });
}
$("#vault-search").addEventListener("input",paintVault);
$("#vault-rx").addEventListener("change",paintVault);
$("#vault-addbtn").addEventListener("click",async()=>{
  const s=$("#vault-add-src").value.trim(), t=$("#vault-add-tgt").value.trim();
  if(!s||!t){ return; }
  await api().approve_term(s,t); $("#vault-add-src").value=""; $("#vault-add-tgt").value="";
  await renderVault();
});
$("#vault-export").addEventListener("click",async()=>{
  const r=await api().export_vault();
  if(r.ok) alert(`Exported ${r.count} term(s) to:\n${r.path}`);
});
$("#vault-import").addEventListener("click",async()=>{
  const r=await api().import_vault();
  if(r.error){ alert("Import failed: "+r.error); return; }
  if(r.ok){ await renderVault(); alert(`Imported ${r.added} term(s). Vault now has ${r.count}.`); }
});
$("#vault-clear").addEventListener("click",async()=>{
  const r0=await api().termbase_info(); if(!r0.count) return;
  if(!confirm(`Clear all ${r0.count} approved term(s)? This cannot be undone.`)) return;
  await api().clear_termbase(); await renderVault();
});

// ---------- local verifier: back-translation | LaBSE ----------
let verifierChoice="labse";
$("#verifier").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{
  $("#verifier").querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); verifierChoice=b.dataset.v;
}));
$("#mtall").addEventListener("click", async ()=>{
  const label=verifierChoice==="labse"?"LaBSE":"Back-translation";
  $("#mtall").disabled=true;
  $("#toolshint").textContent=`Loading ${label} model & verifying (first run downloads the model)…`;
  const r=await api().verify_all(verifierChoice);
  $("#mtall").disabled=false;
  if(r.error){ $("#toolshint").textContent="Verify error: "+r.error; return; }
  if(!r.verdicts.length){ $("#toolshint").textContent="No inconsistent flags to verify."; return; }
  const by=new Map(r.verdicts.map(v=>[v.ngram,v]));
  document.querySelectorAll("#results .group").forEach((g,i)=>{
    const f=lastRendered[i]; if(!f) return; const v=by.get(f.ngram); if(!v) return;
    const out=g.querySelector(".mtout"); if(!out) return;
    out.classList.remove("hidden"); out.classList.toggle("bad", v.verdict==="distinct");
    const rows=v.rows.map(x=>`<span dir="rtl">${esc(x.span)}</span> → ${esc(x.note)}`).join("<br>");
    out.innerHTML=`<b>${label}: ${v.verdict}</b> · ${esc(v.summary)} <span class="hint">(advisory)</span><div class="bt">${rows}</div>`;
  });
  $("#toolshint").textContent=`${label} verified ${r.verdicts.length} flag(s) — advisory only.`;
});

// ---------- LLM: judge all ----------
$("#llmall").addEventListener("click", async ()=>{
  $("#llmall").disabled=true; $("#toolshint").textContent="Asking the model about every flag…";
  const r=await api().llm_judge_all();
  $("#llmall").disabled=false;
  if(r.error){ $("#toolshint").textContent="LLM error: "+r.error; return; }
  const byNgram=new Map(r.verdicts.map(v=>[v.ngram,v]));
  document.querySelectorAll("#results .group").forEach((g,i)=>{
    const f=lastRendered[i]; if(!f) return; const v=byNgram.get(f.ngram); if(!v) return;
    let out=g.querySelector(".llmout"); if(out){ out.classList.remove("hidden","bad");
      out.textContent = v.error ? ("LLM error: "+v.error) : `Verdict: ${v.verdict||"?"}${v.preferred?` · preferred: ${v.preferred}`:''}${v.reason?` — ${v.reason}`:''}`;
      if(v.error||v.verdict==="inconsistent") out.classList.add("bad"); }
  });
  $("#toolshint").textContent=`LLM judged ${r.verdicts.length} flag(s).`;
});

// ---------- why-flagged: codepoint diff ----------
function cpOf(ch){return ch.codePointAt(0).toString(16).toUpperCase().padStart(4,"0");}
function suspicious(ch){
  const c=ch.codePointAt(0);
  if(c<0x20) return "control";
  if(c===0x00A0) return "no-break space";
  if(c===0x0640) return "tatweel";
  if(c>=0x200B&&c<=0x200F) return "zero-width / mark";
  if((c>=0x202A&&c<=0x202E)||(c>=0x2066&&c<=0x2069)) return "bidi control";
  if((c>=0x41&&c<=0x5A)||(c>=0x61&&c<=0x7A)) return "Latin letter";
  if(c>=0x30&&c<=0x39) return "ASCII digit";
  if(c===0x06CC) return "Farsi yeh (ی)";
  if(c===0x06A9) return "Farsi keheh (ک)";
  if(c===0x0649) return "alef maksura";
  return "";
}
function disp(ch){return ch===" "?"␠":esc(ch);}
function cpTag(ch){return `<cite>U+${cpOf(ch)}</cite>`;}
function annotate(s){
  return [...s].map(ch=>{
    const n=suspicious(ch);
    return n ? `<span class="cp" title="U+${cpOf(ch)} ${n}">${disp(ch)}${cpTag(ch)}</span>` : esc(ch);
  }).join("");
}
function diffHtml(a,b){
  const A=[...a], B=[...b], n=A.length, m=B.length;
  const dp=Array.from({length:n+1},()=>new Array(m+1).fill(0));
  for(let i=n-1;i>=0;i--)for(let j=m-1;j>=0;j--)
    dp[i][j]=A[i]===B[j]?dp[i+1][j+1]+1:Math.max(dp[i+1][j],dp[i][j+1]);
  let i=0,j=0,out="";
  const emit=(cls,ch)=>{ const t=suspicious(ch)?cpTag(ch):""; out+=cls?`<span class="${cls}">${disp(ch)}${t}</span>`:disp(ch)+t; };
  while(i<n&&j<m){
    if(A[i]===B[j]){ emit("",A[i]); i++;j++; }
    else if(dp[i+1][j]>=dp[i][j+1]){ emit("d-del",A[i]); i++; }
    else { emit("d-add",B[j]); j++; }
  }
  while(i<n){ emit("d-del",A[i]); i++; }
  while(j<m){ emit("d-add",B[j]); j++; }
  return out;
}
function whyFlagged(f){
  const vs=f.variants;
  let html=`<div class="whyhead">Spans compared as distinct (after normalization):</div>`;
  html+=vs.map((v,i)=>`<div class="whyrow"><span class="whytag">Variant ${i+1} · ${v.count}×</span><span class="whyspan" dir="rtl">${annotate(v.span)}</span></div>`).join("");
  if(vs.length>=2){
    html+=`<div class="whyhead">Character diff — Variant 1 <span style="color:var(--rust)">removed</span> / Variant 2 <span style="color:var(--jade)">added</span>:</div>`;
    html+=`<div class="whydiff" dir="rtl">${diffHtml(vs[0].span,vs[1].span)}</div>`;
    if(vs.length>2) html+=`<div class="hint" style="margin-top:6px">(${vs.length-2} more variant(s); diff shows the top two.)</div>`;
  }
  const seen=new Map();
  vs.forEach(v=>[...v.span].forEach(ch=>{const n=suspicious(ch); if(n&&!seen.has(ch)) seen.set(ch,n);}));
  if(seen.size){
    html+=`<div class="whyhead">Unusual characters present:</div><div class="whylegend">`+
      [...seen].map(([ch,n])=>`<span class="cp">${disp(ch)} <cite>U+${cpOf(ch)}</cite> ${esc(n)}</span>`).join("")+`</div>`;
  }else{
    html+=`<div class="hint" style="margin-top:8px">No unusual characters — the spans differ in the aligned words themselves (an alignment/extraction difference, not a hidden character).</div>`;
  }
  return html;
}

// ---------- utils ----------
function esc(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function cssEsc(s){return (window.CSS&&CSS.escape)?CSS.escape(s):s.replace(/["\\#.:]/g,"\\$&");}
function hl(s,q){const e=esc(s);if(!q)return e;try{return e.replace(new RegExp("("+q.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")+")","ig"),"<mark>$1</mark>");}catch{return e;}}

paintChips();
