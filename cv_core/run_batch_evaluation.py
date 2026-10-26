import os, sys, glob, time, json
sys.path.insert(0, ".")
sys.path.insert(0, "d:/Projects/Vodokanal_CV")
from PIL import Image
from cv_core.pipeline import RVCSafetyPipeline

p = RVCSafetyPipeline()

all_raw_files = sorted(glob.glob("rvk-ai-vision-all-raw-*/raw/*.jpg"))
step = max(1, len(all_raw_files) // 25)
selected_files = all_raw_files[::step][:25]

print(f"Reprocessing {len(selected_files)} frames with v2.3 pipeline...")

out_dir = "dataset/previews/batch_results"
os.makedirs(out_dir, exist_ok=True)

results = []
for idx, fpath in enumerate(selected_files):
    fname = os.path.basename(fpath)
    img = Image.open(fpath).convert("RGB")
    
    t0 = time.perf_counter()
    annotated, events = p.process_frame(img, frame_idx=idx, timestamp=time.time())
    dt_ms = (time.perf_counter() - t0) * 1000
    
    save_name = f"res_{idx:02d}_{fname}"
    annotated.save(os.path.join(out_dir, save_name), quality=88)
    
    active = [t for t in p.tracker.tracks.values() if t.last_frame == idx]
    people_info = []
    for t in active:
        people_info.append({
            "track_id": t.track_id,
            "hardhat": t.smoothed_has_hardhat,
            "vest": t.smoothed_has_vest,
            "bbox": t.bbox
        })
        
    results.append({
        "idx": idx,
        "filename": fname,
        "img_src": f"batch_results/{save_name}",
        "dt_ms": dt_ms,
        "people_count": len(people_info),
        "people": people_info
    })

# HTML with smooth interactive Modal Lightbox with WHEEL ZOOM & PAN
cards_html = ""
for item in results:
    hh_count = sum(1 for p in item["people"] if p["hardhat"])
    nohh_count = sum(1 for p in item["people"] if not p["hardhat"])
    
    cards_html += f"""
    <div class="card">
      <div class="card-header">
        <span class="card-title">Кадр #{item['idx']+1}: {item['filename']}</span>
        <span class="card-meta">Инференс: {item['dt_ms']:.1f} мс</span>
      </div>
      <div class="img-wrap" onclick="openModal({item['idx']})">
        <img src="{item['img_src']}" alt="Processed frame" loading="lazy">
      </div>
      <div class="card-footer">
        <span class="badge badge-people">👤 Людей: {item['people_count']}</span>
        <span class="badge badge-clean">⛑️ В каске: {hh_count}</span>
        <span class="badge badge-violation">❌ Без каски: {nohh_count}</span>
        <span class="badge badge-fps">⚡ {item['dt_ms']:.1f} мс (~{1000/max(item['dt_ms'], 1):.0f} FPS)</span>
      </div>
    </div>
    """

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>РВК Safety: Результаты пайплайна v2.3 (Интерактивный зум колёсиком мыши)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #090d16; color: #f8fafc; margin: 0; padding: 20px; }}
  header {{ background: #151e2e; padding: 16px 24px; border-radius: 12px; border: 1px solid #26354a; margin-bottom: 24px; }}
  h1 {{ color: #38bdf8; font-size: 20px; margin-bottom: 6px; }}
  p {{ color: #94a3b8; font-size: 13px; margin: 0; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; max-width: 1400px; margin: 0 auto; }}
  .card {{ background: #151e2e; border: 1px solid #26354a; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
  .card-header {{ padding: 12px 18px; background: #0f172a; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #26354a; }}
  .card-title {{ font-size: 14px; font-weight: 700; color: #38bdf8; }}
  .card-meta {{ font-size: 12px; color: #94a3b8; }}
  .img-wrap {{ width: 100%; background: black; cursor: pointer; position: relative; }}
  .img-wrap:hover::after {{ content: "🔍 Нажмите для детального просмотра и зума"; position: absolute; bottom: 12px; right: 12px; background: rgba(0,0,0,0.85); color: white; padding: 6px 12px; border-radius: 6px; font-size: 12px; }}
  .img-wrap img {{ width: 100%; display: block; }}
  .card-footer {{ padding: 10px 18px; font-size: 12px; display: flex; gap: 10px; flex-wrap: wrap; background: #151e2e; }}
  .badge {{ padding: 4px 10px; border-radius: 12px; font-weight: 600; }}
  .badge-people {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; }}
  .badge-clean {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; }}
  .badge-violation {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
  .badge-fps {{ background: rgba(168, 85, 247, 0.2); color: #c084fc; }}

  /* Modal Lightbox with Pan & Zoom */
  .modal-overlay {{
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(0, 0, 0, 0.94);
    z-index: 1000;
    flex-direction: column;
  }}
  .modal-overlay.active {{ display: flex; }}
  .modal-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 24px;
    background: #0f172a;
    border-bottom: 1px solid #26354a;
    color: white;
    z-index: 1010;
  }}
  .modal-controls {{ display: flex; gap: 10px; align-items: center; }}
  .modal-btn {{ background: #334155; color: white; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; }}
  .modal-btn:hover {{ background: #475569; }}
  .modal-close {{ background: #ef4444; }}
  .modal-close:hover {{ background: #dc2626; }}
  .zoom-hint {{ font-size: 12px; color: #94a3b8; margin-left: 10px; }}

  .viewport {{
    flex: 1;
    overflow: hidden;
    position: relative;
    cursor: grab;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #040711;
  }}
  .viewport:active {{ cursor: grabbing; }}
  .modal-img {{
    max-width: 90vw;
    max-height: 85vh;
    transform-origin: center center;
    transition: transform 0.05s ease-out;
    user-select: none;
    pointer-events: none;
  }}
</style>
</head>
<body>
  <header>
    <h1>⚡ РВК Safety: Галерея результатов пайплайна v2.3</h1>
    <p>Крупные жирные надписи <strong>Arial Bold 22px</strong> • Отфильтрован стенд • <strong>Зум колёсиком мыши + перетаскивание</strong></p>
  </header>

  <div class="grid">
    {cards_html}
  </div>

  <div class="modal-overlay" id="modal">
    <div class="modal-bar">
      <div class="modal-controls">
        <button class="modal-btn" onclick="prevModal()">← Назад</button>
        <button class="modal-btn" onclick="nextModal()">Вперёд →</button>
        <span id="modal-title" style="margin-left: 10px; font-weight: bold; color: #38bdf8;"></span>
      </div>
      <div class="modal-controls">
        <button class="modal-btn" onclick="changeZoom(0.25)">🔍 +</button>
        <button class="modal-btn" onclick="changeZoom(-0.25)">🔍 -</button>
        <button class="modal-btn" onclick="resetZoom()">100%</button>
        <span class="zoom-hint">Крутите колёсико мыши для зума, зажмите и тяните для панорамирования</span>
        <button class="modal-btn modal-close" onclick="closeModal()">✕ Закрыть (Esc)</button>
      </div>
    </div>
    <div class="viewport" id="viewport">
      <img class="modal-img" id="modal-img" src="" alt="Zoomed frame">
    </div>
  </div>

<script>
const items = {json.dumps(results, ensure_ascii=False)};
let currentIndex = 0;
let scale = 1;
let posX = 0, posY = 0;
let isDragging = false;
let startX = 0, startY = 0;

const viewport = document.getElementById('viewport');
const modalImg = document.getElementById('modal-img');

function updateTransform() {{
  modalImg.style.transform = `translate(${{posX}}px, ${{posY}}px) scale(${{scale}})`;
}}

function resetZoom() {{
  scale = 1;
  posX = 0;
  posY = 0;
  updateTransform();
}}

function changeZoom(delta) {{
  scale = Math.max(0.5, Math.min(5, scale + delta));
  updateTransform();
}}

viewport.addEventListener('wheel', (e) => {{
  e.preventDefault();
  const delta = e.deltaY < 0 ? 0.2 : -0.2;
  changeZoom(delta);
}});

viewport.addEventListener('mousedown', (e) => {{
  isDragging = true;
  startX = e.clientX - posX;
  startY = e.clientY - posY;
}});

window.addEventListener('mousemove', (e) => {{
  if (!isDragging) return;
  posX = e.clientX - startX;
  posY = e.clientY - startY;
  updateTransform();
}});

window.addEventListener('mouseup', () => {{
  isDragging = false;
}});

function openModal(idx) {{
  currentIndex = idx;
  const modal = document.getElementById('modal');
  const title = document.getElementById('modal-title');
  modalImg.src = items[idx].img_src;
  title.innerText = `Кадр #${{idx + 1}} / ${{items.length}}: ${{items[idx].filename}}`;
  resetZoom();
  modal.classList.add('active');
}}

function closeModal() {{
  document.getElementById('modal').classList.remove('active');
}}

function prevModal() {{
  if (currentIndex > 0) openModal(currentIndex - 1);
}}

function nextModal() {{
  if (currentIndex < items.length - 1) openModal(currentIndex + 1);
}}

window.addEventListener('keydown', (e) => {{
  const modal = document.getElementById('modal');
  if (!modal.classList.contains('active')) return;
  if (e.key === 'Escape') closeModal();
  if (e.key === 'ArrowLeft') prevModal();
  if (e.key === 'ArrowRight') nextModal();
}});
</script>
</body>
</html>
"""

with open("dataset/previews/all_pipeline_results.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Updated dataset/previews/all_pipeline_results.html with v2.3 interactive wheel zoom!")
