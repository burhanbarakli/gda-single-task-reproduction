#!/usr/bin/env python3
"""Extract robomimic logs and build the offline GDA experiment report."""
import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "ot-sim2real": "g5_ot-sim2real_seed1_02",
    "MMD": "g5_MMD_seed1_01",
    "cotrain": "g5_cotrain_seed1_01",
    "source_only": "g5_source_only_seed1_01",
    "target_only": "g5_target_only_seed1_01",
}
COLORS = {"ot-sim2real": "#ffb000", "MMD": "#7c83ff", "cotrain": "#35d0a0", "source_only": "#ff6b7a", "target_only": "#62b6ff"}
DEV = {
    "ot-sim2real": {"down": 6, "up": 3, "total": 9, "n": 20},
    "MMD": {"down": 5, "up": 2, "total": 7, "n": 20},
    "cotrain": {"down": 5, "up": 3, "total": 8, "n": 20},
}
DESCRIPTIONS = {
    "ot-sim2real": "Dengesiz optimal taşıma ve DTW eşleştirmesiyle temsil uyarlama.",
    "MMD": "Kaynak ve hedef temsil dağılımlarını MMD yardımcı kaybıyla yaklaştırma.",
    "cotrain": "Aynı kaynak/hedef haklarıyla, uyarlama yardımcı kaybı olmadan ortak eğitim.",
    "source_only": "Yalnız kaynak gösterimleriyle davranış klonlama.",
    "target_only": "Yalnız 10 hedef gösterimle davranış klonlama.",
}


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def find_log(source: Path, run_id: str) -> Path | None:
    patterns = [
        f"experiments/runs/{run_id}/*/logs/log.txt",
        f"runs/{run_id}/*/logs/log.txt",
        f"runs/{run_id.replace('_01', '_reproduction').replace('_02', '_reproduction')}/*/logs/log.txt",
    ]
    found = []
    for pattern in patterns:
        found.extend(source.glob(pattern))
    return sorted(found)[-1] if found else None


def parse_log(path: Path | None) -> list[dict]:
    if not path:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"Train Epoch\s+(\d+)\s*\r?\n(\{.*?\r?\n\})", re.DOTALL)
    rows = []
    for match in pattern.finditer(text):
        try:
            values = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        row = {"epoch": int(match.group(1))}
        row.update({key: value for key, value in values.items() if isinstance(value, (int, float))})
        if "Loss" in row and "total_loss" not in row:
            row["total_loss"] = row["Loss"]
            row["bc_loss"] = row["Loss"]
        rows.append(row)
    deduped = {row["epoch"]: row for row in rows}
    return [deduped[key] for key in sorted(deduped)]


def result_for(source: Path, run_id: str) -> dict:
    candidates = [source / "experiments/audit" / f"{run_id}-result.json", source / "artifacts" / f"{run_id}-result.json"]
    for path in candidates:
        value = read_json(path)
        if value:
            return value
    return {}


def evaluation_state(source: Path) -> dict:
    for path in [source / "experiments/audit/g5-evaluation-queue-v5-state.json", source / "results/final/state.json"]:
        value = read_json(path)
        if value:
            return value
    return {"status": "not_started", "completed": []}


def collect(source: Path) -> dict:
    checkpoints = read_json(ROOT / "manifests/checkpoints.json", {"checkpoints": []})["checkpoints"]
    checkpoint_by_method = {row["method"]: row for row in checkpoints}
    methods = []
    curves = {}
    for method, run_id in RUNS.items():
        log = find_log(source, run_id)
        rows = parse_log(log)
        result = result_for(source, run_id)
        seconds = result.get("seconds") or result.get("seconds_this_attempt")
        methods.append({
            "method": method,
            "run_id": run_id,
            "description": DESCRIPTIONS[method],
            "color": COLORS[method],
            "status": result.get("status", "completed" if len(rows) >= 500 else "unknown"),
            "epochs": max((row["epoch"] for row in rows), default=0),
            "seconds": seconds,
            "hours": round(seconds / 3600, 2) if seconds else None,
            "gpu_allocated_mib": round(result.get("gpu_peak_allocated_bytes", 0) / 2**20),
            "gpu_reserved_mib": round(result.get("gpu_peak_reserved_bytes", 0) / 2**20),
            "log": str(log) if log else None,
            "final": rows[-1] if rows else {},
            "checkpoint": checkpoint_by_method.get(method),
            "development": DEV.get(method),
        })
        curves[method] = rows
    completed = sum(item["status"] == "completed" and item["epochs"] >= 500 for item in methods)
    total_seconds = sum(item["seconds"] or 0 for item in methods)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source.resolve()),
        "summary": {"completed_methods": completed, "updates": completed * 150000, "training_hours": round(total_seconds / 3600, 2), "training_days": round(total_seconds / 86400, 2)},
        "methods": methods,
        "curves": curves,
        "evaluation": evaluation_state(source),
        "protocol": read_json(ROOT / "manifests/protocol.json", {}),
    }


def write_svg_figures(data: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    width, height, left, top, right, bottom = 1200, 650, 90, 45, 30, 70
    values = [row["total_loss"] for rows in data["curves"].values() for row in rows if row.get("total_loss", 0) > 0]
    low, high = min(values), max(values)
    y0, y1 = math.log10(low), math.log10(high)
    def px(epoch): return left + (width-left-right) * (epoch-1) / 499
    def py(value): return top + (height-top-bottom) * (1-(math.log10(value)-y0)/(y1-y0))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#081018"/><style>text{{font-family:Arial,sans-serif;fill:#dcebf1}}.muted{{fill:#93a9b5}}</style><text x="{left}" y="28" font-size="20" font-weight="700">GDA eğitim loss karşılaştırması</text>']
    for index in range(6):
        y = top + (height-top-bottom)*index/5
        value = 10 ** (y1 + (y0-y1)*index/5)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#27404d"/><text class="muted" x="8" y="{y+4:.1f}" font-size="12">{value:.2e}</text>')
    for epoch in [1,100,200,300,400,500]:
        parts.append(f'<text class="muted" x="{px(epoch)-9:.1f}" y="{height-25}" font-size="12">{epoch}</text>')
    for index, method in enumerate(data["methods"]):
        rows = [row for row in data["curves"][method["method"]] if row.get("total_loss", 0) > 0]
        points = " ".join(f'{px(row["epoch"]):.1f},{py(row["total_loss"]):.1f}' for row in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{method["color"]}" stroke-width="2"/><line x1="{left+index*205}" y1="{height-50}" x2="{left+index*205+24}" y2="{height-50}" stroke="{method["color"]}" stroke-width="4"/><text x="{left+index*205+31}" y="{height-45}" font-size="13">{method["method"]}</text>')
    parts.append('</svg>')
    (directory / "total-loss.svg").write_text("".join(parts), encoding="utf-8")

    dev = [method for method in data["methods"] if method["development"]]
    bars = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="480" viewBox="0 0 900 480"><rect width="100%" height="100%" fill="#081018"/><style>text{{font-family:Arial,sans-serif;fill:#dcebf1}}</style><text x="70" y="34" font-size="20" font-weight="700">Epoch 500 geliştirme rollout başarısı</text>']
    for i, method in enumerate(dev):
        x, rate = 150+i*240, method["development"]["total"]/method["development"]["n"]
        h = 330*rate
        bars.append(f'<rect x="{x}" y="{410-h:.1f}" width="130" height="{h:.1f}" rx="8" fill="{method["color"]}"/><text x="{x+42}" y="{395-h:.1f}" font-size="22" font-weight="700">{rate*100:.0f}%</text><text x="{x}" y="442" font-size="15">{method["method"]}</text>')
    bars.append('<text x="70" y="468" fill="#93a9b5" font-size="12">10 down + 10 up rollout; nihai karşılaştırma değildir.</text></svg>')
    (directory / "development-success.svg").write_text("".join(bars), encoding="utf-8")


HTML = r'''<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDA · Beş Eğitim Deneyi</title>
<style>
:root{--bg:#081018;--panel:#101c27;--panel2:#152533;--ink:#edf6fa;--muted:#93a9b5;--line:#27404d;--accent:#ffb000;--ok:#35d0a0;--warn:#ffcf5c}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#18384b 0,transparent 30%),var(--bg);color:var(--ink);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}main{max-width:1460px;margin:auto;padding:30px}.hero{display:flex;justify-content:space-between;gap:30px;align-items:end;border-bottom:1px solid var(--line);padding:22px 0 28px}.eyebrow{color:var(--accent);font-weight:800;letter-spacing:.15em;text-transform:uppercase;font-size:12px}h1{font-size:clamp(32px,5vw,68px);line-height:1;margin:10px 0 14px;max-width:900px}.lede{color:var(--muted);max-width:850px;font-size:17px}.stamp{text-align:right;color:var(--muted);min-width:240px}.grid{display:grid;gap:16px}.stats{grid-template-columns:repeat(5,1fr);margin:24px 0}.card,.panel{background:linear-gradient(145deg,rgba(21,37,51,.95),rgba(12,24,34,.95));border:1px solid var(--line);border-radius:14px}.card{padding:18px}.card b{display:block;font-size:27px}.card span{color:var(--muted)}.panel{padding:22px;margin:18px 0}h2{font-size:24px;margin:0 0 5px}h3{font-size:16px;margin:0}.sub{color:var(--muted);margin:0 0 18px}.toolbar{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}select,button{background:#0b1720;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:9px 12px}label.check{border:1px solid var(--line);border-radius:20px;padding:6px 10px;color:var(--muted)}canvas{width:100%;height:420px;background:#09141d;border-radius:10px}.two{grid-template-columns:1.55fr 1fr}.methods{grid-template-columns:repeat(5,1fr)}.method{padding:16px;border-top:4px solid var(--color)}.method .value{font-size:22px;font-weight:800}.method p{color:var(--muted);min-height:68px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:11px 9px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;word-break:break-all}.badge{display:inline-block;padding:4px 9px;border-radius:20px;background:#193a35;color:#8ff5d4;font-weight:700}.badge.wait{background:#493a17;color:#ffe29b}.bar{height:11px;background:#071119;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:var(--color)}.videos{grid-template-columns:repeat(3,1fr)}video{width:100%;border-radius:9px;background:#000}.note{padding:13px;border-left:3px solid var(--accent);background:#101921;color:var(--muted)}details{border-bottom:1px solid var(--line);padding:11px 0}summary{cursor:pointer;font-weight:700}.footer{color:var(--muted);font-size:13px;padding:25px 0 60px}@media(max-width:1050px){.stats,.methods{grid-template-columns:repeat(2,1fr)}.two,.videos{grid-template-columns:1fr}.hero{display:block}.stamp{text-align:left;margin-top:20px}}@media(max-width:600px){main{padding:16px}.stats,.methods{grid-template-columns:1fr}canvas{height:320px}}
</style></head><body><main>
<section class="hero"><div><div class="eyebrow">Generalizable Domain Adaptation · G5</div><h1>Beş eğitimin deney kaydı</h1><p class="lede">Stack_RL2_range görevinde rgb → table-wood görsel alan değişimi. Tek tohum, eşit 150.000 güncelleme bütçesi ve Diffusion Policy mimarisi.</p></div><div class="stamp"><b id="generated"></b><br>Çevrimdışı, taşınabilir rapor<br><span id="liveFlag"></span></div></section>
<section class="grid stats"><div class="card"><b id="complete">—</b><span>tamamlanan yöntem</span></div><div class="card"><b id="updates">—</b><span>toplam güncelleme</span></div><div class="card"><b id="hours">—</b><span>GPU eğitim saati</span></div><div class="card"><b>500</b><span>epoch / yöntem</span></div><div class="card"><b>1</b><span>donmuş eğitim tohumu</span></div></section>
<section class="panel"><h2>Canlı operasyon durumu</h2><p class="sub">Eğitim tamamlandı; nihai 500-rollout test kuyruğunun son anlık görüntüsü.</p><div id="evalStatus"></div></section>
<section class="grid methods" id="methodCards"></section>
<section class="panel"><h2>Loss karşılaştırması</h2><p class="sub">Aynı eksende yöntem seçin. Farklı yardımcı loss tanımları aynı sayısal ölçeği paylaşmaz; performans hükmü için rollout kullanılır.</p><div class="toolbar"><select id="metric"></select><select id="smooth"><option value="1">Ham</option><option value="5">5 epoch ortalama</option><option value="15">15 epoch ortalama</option></select><span id="checks"></span></div><canvas id="chart"></canvas><p class="note" id="lossNote"></p></section>
<section class="grid two"><div class="panel"><h2>Epoch 500 geliştirme kontrolü</h2><p class="sub">Her görünümde 10 rollout. Küçük geliştirme testi; nihai sıralama değildir.</p><div id="devTable"></div></div><div class="panel"><h2>Süre ve GPU belleği</h2><p class="sub">Wall-clock süre kesinti/devam etkisini içerir.</p><div id="resourceTable"></div></div></section>
<section class="panel"><h2>Model kayıtları</h2><p class="sub">İkili checkpointler Git dışında tutulur; bu kayıtlar dosyayı kesin biçimde tanımlar.</p><div id="checkpointTable"></div></section>
<section class="panel"><h2>Ortak eğitim parametreleri</h2><div id="params"></div></section>
<section class="panel"><h2>Görsel materyaller</h2><p class="sub">Düzenlenebilir vektör grafikler: <a href="assets/total-loss.svg" style="color:#62b6ff">total-loss.svg</a> · <a href="assets/development-success.svg" style="color:#62b6ff">development-success.svg</a>. Aşağıda epoch 500 geliştirme rolloutlarından örnek videolar bulunur; tek video başarı oranını temsil etmez.</p><div class="grid videos" id="videos"></div></section>
<section class="panel"><h2>Yorumlama sınırları</h2><details open><summary>Bu deney neyi gösteriyor?</summary><p>Resmî yöntemin tek görev ve tek tohum kontrollü yeniden üretimini, dört temel karşılaştırmayla gösteriyor.</p></details><details><summary>Neyi göstermiyor?</summary><p>Fiziksel robot aktarımı, çok görev genellemesi ve tohumlar arası belirsizlik bu aşamada ölçülmedi.</p></details><details><summary>OT ile MMD loss farkı nasıl okunmalı?</summary><p>OT total loss, ağırlıklı taşıma terimiyle; MMD total loss ise MMD yardımcı terimiyle kurulur. Ham büyüklük farkı tek başına bir yöntemin daha iyi olduğunu söylemez.</p></details></section>
<p class="footer">Kaynak commit: <span class="mono">114704b6b381b410cc14a51cb16952d1f4e8c69d</span> · Rapor verisi: <span id="source" class="mono"></span></p>
</main><script>
let DATA=__DATA__; const fmt=n=>new Intl.NumberFormat('tr-TR',{maximumFractionDigits:2}).format(n||0); const esc=s=>String(s??'—').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function render(){document.querySelector('#generated').textContent=new Date(DATA.generated_at).toLocaleString('tr-TR');document.querySelector('#complete').textContent=DATA.summary.completed_methods+'/5';document.querySelector('#updates').textContent=fmt(DATA.summary.updates);document.querySelector('#hours').textContent=fmt(DATA.summary.training_hours);document.querySelector('#source').textContent=DATA.source_root;
 const ev=DATA.evaluation||{};let badge=ev.status==='completed'?'badge':'badge wait';document.querySelector('#evalStatus').innerHTML=`<span class="${badge}">${esc(ev.status)}</span> &nbsp; Tamamlanan: ${esc((ev.completed||[]).length)}/5 &nbsp; Güncelleme: ${esc(ev.updated_at||'—')}<p class="sub">Bekleyen/çalışan yöntem: ${esc(ev.current_run||'—')}. Donmuş plan: 50 down + 50 up rollout / yöntem.</p>`;
 document.querySelector('#methodCards').innerHTML=DATA.methods.map(m=>`<article class="card method" style="--color:${m.color}"><h3>${esc(m.method)}</h3><div class="value">${m.epochs}/500 epoch</div><p>${esc(m.description)}</p><span class="badge">${esc(m.status)}</span></article>`).join('');
 const keys=[...new Set(Object.values(DATA.curves).flat().flatMap(Object.keys))].filter(k=>!['epoch'].includes(k)); const preferred=['total_loss','bc_loss','ot_loss','MMD_loss','Time_Epoch','Time_Data_Loading'];keys.sort((a,b)=>(preferred.indexOf(a)<0?99:preferred.indexOf(a))-(preferred.indexOf(b)<0?99:preferred.indexOf(b))); document.querySelector('#metric').innerHTML=keys.map(k=>`<option>${esc(k)}</option>`).join('');
 document.querySelector('#checks').innerHTML=DATA.methods.map(m=>`<label class="check"><input type="checkbox" value="${m.method}" checked> <span style="color:${m.color}">●</span> ${m.method}</label>`).join('');
 document.querySelector('#devTable').innerHTML='<table><tr><th>Yöntem</th><th>Down</th><th>Up</th><th>Toplam</th><th>Oran</th></tr>'+DATA.methods.map(m=>{let d=m.development;return `<tr><td>${m.method}</td><td>${d?d.down+'/10':'—'}</td><td>${d?d.up+'/10':'—'}</td><td>${d?d.total+'/'+d.n:'—'}</td><td style="min-width:120px">${d?`<div class="bar" style="--color:${m.color}"><i style="width:${d.total/d.n*100}%"></i></div> ${fmt(d.total/d.n*100)}%`:'yapılmadı'}</td></tr>`}).join('')+'</table>';
 document.querySelector('#resourceTable').innerHTML='<table><tr><th>Yöntem</th><th>Saat</th><th>Allocated</th><th>Reserved</th></tr>'+DATA.methods.map(m=>`<tr><td>${m.method}</td><td>${fmt(m.hours)}</td><td>${fmt(m.gpu_allocated_mib)} MiB</td><td>${fmt(m.gpu_reserved_mib)} MiB</td></tr>`).join('')+'</table>';
 document.querySelector('#checkpointTable').innerHTML='<table><tr><th>Yöntem / koşu</th><th>Dosya</th><th>Boyut</th><th>SHA-256</th></tr>'+DATA.methods.map(m=>{let c=m.checkpoint||{};return `<tr><td><b>${m.method}</b><br><span class="mono">${m.run_id}</span></td><td>${esc(c.file)}</td><td>${c.bytes?fmt(c.bytes/2**30)+' GiB':'—'}</td><td class="mono">${esc(c.sha256)}</td></tr>`}).join('')+'</table>';
 const p=DATA.protocol.training||{};document.querySelector('#params').innerHTML=`<table><tr><th>Görev</th><td>${esc(DATA.protocol.task)}</td><th>Alan</th><td>rgb → ${esc(DATA.protocol.domain_shift)}</td></tr><tr><th>Epoch / adım</th><td>${p.epochs} × ${p.steps_per_epoch}</td><th>Batch / worker</th><td>${p.batch_size} / ${p.workers}</td></tr><tr><th>Sequence / frame stack</th><td>${p.sequence_length} / ${p.frame_stack}</td><th>Checkpoint</th><td>Her ${p.checkpoint_every_epochs} epoch</td></tr><tr><th>Mimari</th><td colspan="3">${p.algorithm}, ${p.encoder} encoder, ${p.denoiser} denoiser</td></tr></table>`;
 const vids=[['OT-Sim2Real','ot-down.mp4','ot-up.mp4'],['MMD','mmd-down.mp4','mmd-up.mp4'],['Co-training','cotrain-down.mp4','cotrain-up.mp4']];document.querySelector('#videos').innerHTML=vids.map(v=>`<article><h3>${v[0]} · down</h3><video controls muted preload="metadata" src="assets/videos/${v[1]}"></video><h3>${v[0]} · up</h3><video controls muted preload="metadata" src="assets/videos/${v[2]}"></video></article>`).join(''); draw();}
function smooth(rows,key,n){return rows.map((r,i)=>{let a=rows.slice(Math.max(0,i-n+1),i+1).map(x=>x[key]).filter(Number.isFinite);return {epoch:r.epoch,value:a.length?a.reduce((x,y)=>x+y,0)/a.length:null}})}
function draw(){const c=document.querySelector('#chart'),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,h);const key=document.querySelector('#metric').value||'total_loss',n=+document.querySelector('#smooth').value;const chosen=[...document.querySelectorAll('#checks input:checked')].map(e=>e.value);let sets=DATA.methods.filter(m=>chosen.includes(m.method)).map(m=>({m,rows:smooth(DATA.curves[m.method]||[],key,n).filter(r=>Number.isFinite(r.value)&&r.value>0)})).filter(s=>s.rows.length);let vals=sets.flatMap(s=>s.rows.map(r=>r.value));if(!vals.length){x.fillStyle='#93a9b5';x.fillText('Bu metrik için veri yok',30,40);return}let log=key.includes('loss')||key==='total_loss'||key==='bc_loss',min=Math.min(...vals),max=Math.max(...vals);if(min===max){min*=.9;max*=1.1}const tx=e=>55+(w-75)*(e-1)/499,ty=v=>{let a=log?Math.log10(min):min,b=log?Math.log10(max):max,q=log?Math.log10(v):v;return 20+(h-60)*(1-(q-a)/(b-a))};x.strokeStyle='#27404d';x.fillStyle='#93a9b5';x.font='12px Segoe UI';for(let i=0;i<6;i++){let yy=20+(h-60)*i/5;x.beginPath();x.moveTo(55,yy);x.lineTo(w-20,yy);x.stroke();let a=log?Math.log10(max)+(Math.log10(min)-Math.log10(max))*i/5:max+(min-max)*i/5,val=log?10**a:a;x.fillText(val.toExponential(2),4,yy+4)}for(let e of [1,100,200,300,400,500])x.fillText(e,tx(e)-8,h-16);for(let s of sets){x.strokeStyle=s.m.color;x.lineWidth=2;x.beginPath();s.rows.forEach((r,i)=>{let X=tx(r.epoch),Y=ty(r.value);i?x.lineTo(X,Y):x.moveTo(X,Y)});x.stroke()}document.querySelector('#lossNote').textContent=`${key} · ${n===1?'ham değer':n+' epoch hareketli ortalama'} · ${log?'logaritmik':'doğrusal'} Y ekseni.`}
document.addEventListener('change',e=>{if(e.target.closest('.toolbar'))draw()});addEventListener('resize',draw);render();
if(location.protocol.startsWith('http')){document.querySelector('#liveFlag').textContent='Canlı sunucu: 15 sn yenileme';setInterval(async()=>{try{let r=await fetch('/api/data');if(r.ok){DATA=await r.json();render()}}catch(e){}},15000)}
</script></body></html>'''


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML.replace("__DATA__", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/gda-training-report.html")
    args = parser.parse_args()
    data = collect(args.source_root.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(data), encoding="utf-8")
    (args.output.parent / "data/training-summary.json").write_text(json.dumps({key: value for key, value in data.items() if key != "curves"}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output.parent / "data/loss-curves.json").write_text(json.dumps(data["curves"], ensure_ascii=False), encoding="utf-8")
    write_svg_figures(data, args.output.parent / "assets")
    print(json.dumps({"output": str(args.output.resolve()), "methods": len(data["methods"]), "completed": data["summary"]["completed_methods"], "curve_points": sum(map(len, data["curves"].values()))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
