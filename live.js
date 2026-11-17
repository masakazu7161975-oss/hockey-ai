"use strict";
/* =====================================================================
   ライブ解析（試合中モード・試作）
   - 三脚で固定したカメラ映像（または録画した動画）から YOLO で人物を検出
   - ユニフォームの色で 自チーム／相手／除外（審判・PB・GK など）に分類
   - 画面上でタップしたピッチ四隅から射影変換し、フィールド座標（m）に変換
   - AI が数える: 23m侵入、サークル侵入、相手陣でプレーした時間
   - 人がタップ: ボール保持、シュート（枠内／枠外／ゴール）、PC、PS
   - AI の結果は source:"ai", status:"estimated" で保存（試合後にコーチが確認）
   すべて端末内で処理し、映像は外部に送信しない
   ===================================================================== */

const PITCH = { L: 91.4, W: 55.0, D23: 22.9, R: 14.63, POST: 1.83 };
const IN_W = 1280, IN_H = 480;           // モデル入力（横長のピッチに合わせた形）
const CAT = { own: "自チーム", opp: "相手", ex: "除外" };

const live = {
  stream: null, running: false, loopId: 0, sess: null, backend: "", ms: 0,
  cfg: null, q: null, dets: [], lastT: 0, busy: false,
};

/* ---------- 設定（端末内に保存） */
function liveCfg() {
  if (!live.cfg) live.cfg = prefs.get("liveCfg", { corners: [], samples: [], ownAttack: "right" });
  return live.cfg;
}
function saveCfg() { prefs.set("liveCfg", live.cfg); }

/* ---------- 射影変換（画面 → フィールド m） */
function solveH(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  for (let c = 0; c < 8; c++) {           // ガウスの消去法
    let p = c; for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 8; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  return b.map((v, i) => v / A[i][i]);
}
const applyH = (H, x, y) => { const w = H[6] * x + H[7] * y + 1; return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w]; };
// 四隅は画面上で 左奥・右奥・右手前・左手前 の順にタップ（左ゴール側が x=0）
const FIELD_CORNERS = [[0, 0], [PITCH.L, 0], [PITCH.L, PITCH.W], [0, PITCH.W]];
const homography = () => { const c = liveCfg().corners; return c.length === 4 ? solveH(c, FIELD_CORNERS) : null; };
const invHomography = () => { const c = liveCfg().corners; return c.length === 4 ? solveH(FIELD_CORNERS, c) : null; };

/* ---------- ゾーン判定 */
function inZone(x, y, goalX, zone) {
  if (zone === "23") return Math.abs(x - goalX) < PITCH.D23;
  // サークル: ゴールポスト間の線分からの距離が 14.63m 以内
  const cy = PITCH.W / 2, yy = Math.max(cy - PITCH.POST, Math.min(cy + PITCH.POST, y));
  return Math.hypot(x - goalX, y - yy) <= PITCH.R;
}
// その Q で各チームが攻めるゴールの x 座標（後半はエンドが入れ替わる）
function attackGoal(team, quarter) {
  const swap = quarter >= 3;
  const ownRight = (liveCfg().ownAttack === "right") !== swap;
  const ownGoal = ownRight ? PITCH.L : 0;
  return team === "own" ? ownGoal : PITCH.L - ownGoal;
}

/* ---------- 色（Lab）と分類 */
function rgb2lab(r, g, b) {
  const f = c => { c /= 255; return c > 0.04045 ? ((c + 0.055) / 1.055) ** 2.4 : c / 12.92; };
  r = f(r); g = f(g); b = f(b);
  let x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047, y = r * 0.2126 + g * 0.7152 + b * 0.0722, z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
  const t = v => v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116;
  x = t(x); y = t(y); z = t(z);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}
// 胴体部分の平均色（青い人工芝の画素は除く）
function torsoLab(ctx, d) {
  const w = d.x2 - d.x1, h = d.y2 - d.y1;
  const x = Math.round(d.x1 + w * 0.25), y = Math.round(d.y1 + h * 0.18), cw = Math.max(1, Math.round(w * 0.5)), ch = Math.max(1, Math.round(h * 0.32));
  let px; try { px = ctx.getImageData(x, y, cw, ch).data; } catch { return null; }
  let R = 0, G = 0, B = 0, n = 0;
  for (let i = 0; i < px.length; i += 4) {
    const r = px[i], g = px[i + 1], b = px[i + 2];
    if (b > r + 35 && b > g + 10) continue; // 芝（青）
    R += r; G += g; B += b; n++;
  }
  if (n < 6) return null;
  return rgb2lab(R / n, G / n, B / n);
}
function classify(lab) {
  if (!lab) return null;
  let best = null, bd = 1e9;
  for (const s of liveCfg().samples) {
    const d = Math.hypot(lab[0] - s.lab[0], lab[1] - s.lab[1], lab[2] - s.lab[2]);
    if (d < bd) { bd = d; best = s.cat; }
  }
  return bd < 40 ? best : null;
}

/* ---------- 物体検出（YOLO / ONNX Runtime Web） */
function loadScript(src) {
  return new Promise((res, rej) => { if (window.ort) return res(); const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error(src + " を読み込めません")); document.head.append(s); });
}
async function loadModel(status) {
  if (live.sess) return live.sess;
  status("AI モデルを読み込み中…（初回は約30MB。2回目からは端末内から読み込みます）");
  await loadScript("ort/ort.webgpu.min.js");
  ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
  ort.env.wasm.numThreads = 1;
  const tries = navigator.gpu ? [["webgpu"], ["wasm"]] : [["wasm"]];
  let err;
  for (const ep of tries) {
    try { live.sess = await ort.InferenceSession.create("models/yolo11n.onnx", { executionProviders: ep, graphOptimizationLevel: "all" }); live.backend = ep[0]; break; }
    catch (e) { err = e; }
  }
  if (!live.sess) throw err || new Error("AI モデルを起動できません");
  return live.sess;
}
const lb = document.createElement("canvas"); lb.width = IN_W; lb.height = IN_H;
const lbx = lb.getContext("2d", { willReadFrequently: true });
const full = document.createElement("canvas");
const fullx = full.getContext("2d", { willReadFrequently: true });

// ピッチ周辺を切り出して 1280×480 に収め、人物を検出する
async function detect(src, vw, vh) {
  const c = liveCfg().corners;
  let x0 = 0, y0 = 0, x1 = vw, y1 = vh;
  if (c.length === 4) {
    const xs = c.map(p => p[0]), ys = c.map(p => p[1]);
    const mh = (Math.max(...ys) - Math.min(...ys)) * 0.15;
    x0 = Math.max(0, Math.min(...xs) - 20); x1 = Math.min(vw, Math.max(...xs) + 20);
    y0 = Math.max(0, Math.min(...ys) - mh - 40); y1 = Math.min(vh, Math.max(...ys) + 20);
  }
  const cw = x1 - x0, ch = y1 - y0, s = Math.min(IN_W / cw, IN_H / ch);
  const dw = cw * s, dh = ch * s, ox = (IN_W - dw) / 2, oy = (IN_H - dh) / 2;
  lbx.fillStyle = "#727272"; lbx.fillRect(0, 0, IN_W, IN_H);
  lbx.drawImage(src, x0, y0, cw, ch, ox, oy, dw, dh);
  const px = lbx.getImageData(0, 0, IN_W, IN_H).data, N = IN_W * IN_H;
  const input = new Float32Array(3 * N);
  for (let i = 0, j = 0; i < N; i++, j += 4) { input[i] = px[j] / 255; input[i + N] = px[j + 1] / 255; input[i + 2 * N] = px[j + 2] / 255; }
  const t0 = performance.now();
  const out = await live.sess.run({ [live.sess.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, IN_H, IN_W]) });
  live.ms = Math.round(performance.now() - t0);
  const o = out[live.sess.outputNames[0]], A = o.data, K = o.dims[2];
  const boxes = [];
  for (let i = 0; i < K; i++) {
    const sc = A[4 * K + i]; // クラス0 = person
    if (sc < 0.25) continue;
    const cx = A[i], cy = A[K + i], w = A[2 * K + i], h = A[3 * K + i];
    boxes.push({ s: sc, x1: (cx - w / 2 - ox) / s + x0, y1: (cy - h / 2 - oy) / s + y0, x2: (cx + w / 2 - ox) / s + x0, y2: (cy + h / 2 - oy) / s + y0 });
  }
  boxes.sort((a, b) => b.s - a.s);
  const keep = [];
  for (const b of boxes) {
    if (keep.every(k => iou(k, b) < 0.5)) keep.push(b);
    if (keep.length >= 60) break;
  }
  return keep;
}
function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)), iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const i = ix * iy; return i / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i);
}
// 検出結果にフィールド座標とチームを付ける
function annotate(dets) {
  const H = homography();
  for (const d of dets) {
    d.lab = torsoLab(fullx, d);
    d.cat = classify(d.lab);
    if (H) {
      const [fx, fy] = applyH(H, (d.x1 + d.x2) / 2, d.y2);
      d.fx = fx; d.fy = fy;
      d.onPitch = fx > -2 && fx < PITCH.L + 2 && fy > -2 && fy < PITCH.W + 2;
    } else d.onPitch = true;
  }
  return dets;
}

/* ---------- クォーターの記録 */
function newQuarter(n) {
  return {
    n, running: false, t0: 0, elapsed: 0, lastTick: 0,
    poss: { own: 0, opp: 0 }, holder: null,
    terr: { own: 0, opp: 0 },
    zone: { own: { "23": z(), circle: z() }, opp: { "23": z(), circle: z() } },
    counts: { own: c(), opp: c() },
  };
  function z() { return { active: false, since: null, gone: null }; }
  function c() { return { "23m侵入": 0, "サークル侵入": 0, "シュート枠内": 0, "シュート枠外": 0, "ゴール": 0, PC: 0, PS: 0 }; }
}
const qTime = () => { const q = live.q; return q ? q.elapsed + (q.running ? (performance.now() - q.t0) / 1000 : 0) : 0; };
function tickPossession() {
  const q = live.q; if (!q || !q.running) return;
  const now = performance.now(), dt = (now - (q.lastTick || now)) / 1000; q.lastTick = now;
  if (q.holder && dt > 0 && dt < 5) q.poss[q.holder] += dt;
}
async function logTag(team, eventType, result, ai) {
  const m = match(state.matchId); if (!m || !live.q) return;
  const teamId = team === "own" ? m.teamId : m.oppId;
  await DB.put("tags", { matchId: m.id, time: Math.round(qTime() * 10) / 10, quarter: live.q.n, eventType, teamId, playerId: null, result: result || null,
    memo: ai ? "ライブAI" : "ライブ", live: true, source: ai ? "ai" : "manual", status: ai ? "estimated" : "confirmed", createdAt: now(), updatedAt: now() });
  S.tags = await DB.all("tags");
}
async function saveQuarter() {
  const m = match(state.matchId), q = live.q; if (!m || !q) return;
  const rec = (await DB.get("meta", "live:" + m.id)) || { id: "live:" + m.id, quarters: {} };
  rec.quarters[q.n] = { poss: q.poss, terr: q.terr, counts: q.counts, seconds: qTime(), savedAt: now() };
  await DB.put("meta", rec);
}
// ゾーン侵入: 1秒以上いたら「侵入」1回、3秒以上いなくなったら終了
function updateZones(dets, t) {
  const q = live.q;
  for (const team of ["own", "opp"]) {
    const gx = attackGoal(team, q.n);
    for (const zone of ["23", "circle"]) {
      const present = dets.some(d => d.cat === team && d.onPitch && d.fx != null && inZone(d.fx, d.fy, gx, zone));
      const st = q.zone[team][zone];
      if (!st.active) {
        if (present) { st.since ??= t; if (t - st.since >= 1) { st.active = true; st.gone = null; const k = zone === "23" ? "23m侵入" : "サークル侵入"; q.counts[team][k]++; logTag(team, k, null, true); } }
        else st.since = null;
      } else if (!present) { st.gone ??= t; if (t - st.gone >= 3) { st.active = false; st.since = null; } }
      else st.gone = null;
    }
  }
  // 陣地: 選手の平均位置が相手陣側にあるチームに時間を加算
  const xs = dets.filter(d => (d.cat === "own" || d.cat === "opp") && d.onPitch && d.fx != null).map(d => d.fx);
  const dt = Math.min(1, t - (live.lastT || t)); live.lastT = t;
  if (xs.length >= 4 && dt > 0) {
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const ownTowards = attackGoal("own", q.n) > PITCH.L / 2 ? mx > PITCH.L / 2 : mx < PITCH.L / 2;
    q.terr[ownTowards ? "own" : "opp"] += dt;
  }
}

/* ---------- 画面 */
views.live = async () => {
  const cfg = liveCfg();
  const ms = [...S.matches].sort((a, b) => b.date.localeCompare(a.date));
  const wrap = h("div", { class: "stack" });
  if (!ms.length) { wrap.append(h("h2", null, "ライブ解析"), h("div", { class: "panel empty" }, "先に「試合」画面で試合を登録してください。")); return wrap; }
  let m = match(state.matchId) || ms[0];
  if (m.id !== state.matchId) setMatch(m.id);
  if (!live.q || live.q.matchId !== m.id) { live.q = newQuarter(live.q?.n || 1); live.q.matchId = m.id; }
  const ownName = team(m.teamId)?.name ?? "自チーム", oppName = team(m.oppId)?.name ?? "相手";

  const status = h("div", { class: "small muted", id: "lv_status" }, "");
  const say = t => { status.textContent = t; };
  wrap.append(
    h("div", { class: "row spread" }, h("h2", null, "ライブ解析（試作）"), h("span", { class: "pill ai" }, "AI の数は推定値です")),
    h("div", { class: "row" },
      h("label", { class: "field grow" }, "試合", select("lv_match", ms.map(x => [x.id, mLabel(x)]), m.id)),
      h("label", { class: "field" }, "映像", select("lv_src", [["camera", "カメラ（ライブ）"], ["file", "録画した動画で試す"]], prefs.get("lvSrc", "camera")))));
  wrap.querySelector("#lv_match").addEventListener("change", e => { stopLive(); setMatch(+e.target.value); live.q = null; render(); });
  wrap.querySelector("#lv_src").addEventListener("change", e => { prefs.set("lvSrc", e.target.value); stopLive(); render(); });

  const grid = h("div", { class: "tagging live-grid" }), left = h("div", { class: "stack" }), side = h("div", { class: "side live-side" });
  grid.append(left, side); wrap.append(grid);

  /* 映像とオーバーレイ */
  const video = h("video", { playsinline: true, muted: true, autoplay: true });
  video.setAttribute("playsinline", ""); video.muted = true;
  const ov = h("canvas", { class: "lv-ov" });
  const stage = h("div", { class: "player-box lv-stage" }, video, ov);
  left.append(stage, status);
  const srcKind = prefs.get("lvSrc", "camera");
  const startRow = h("div", { class: "row" });
  left.append(startRow);
  if (srcKind === "camera") {
    const camSel = h("select", { id: "lv_cam", style: "width:auto" });
    startRow.append(h("button", { class: "btn primary", onclick: () => startCamera(camSel.value) }, "カメラを開始"), camSel);
    try { (await navigator.mediaDevices?.enumerateDevices?.() || []).filter(d => d.kind === "videoinput").forEach((d, i) => camSel.append(h("option", { value: d.deviceId, selected: d.deviceId === prefs.get("lvCam", "") }, d.label || `カメラ ${i + 1}`))); } catch { }
    if (!camSel.options.length) camSel.append(h("option", { value: "" }, "背面カメラ"));
  } else {
    startRow.append(fileButton("動画を選ぶ", "image/*,video/*", "primary", f => { stopLive(); video.srcObject = null; video.src = URL.createObjectURL(f); video.loop = false; video.controls = true; video.play().catch(() => { }); say("動画を再生しています。下の手順で設定してください。"); }, "lv_file"));
  }

  async function startCamera(deviceId) {
    stopLive();
    try {
      live.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { deviceId: deviceId ? { exact: deviceId } : undefined, facingMode: deviceId ? undefined : "environment", width: { ideal: 1920 }, height: { ideal: 1080 } } });
      video.srcObject = live.stream; await video.play();
      if (deviceId) prefs.set("lvCam", deviceId);
      say(`カメラ: ${video.videoWidth}×${video.videoHeight}。超広角（0.5倍）を使う場合は、上の一覧から「超広角」を選んでください。`);
      render(); // カメラ名（許可後に表示される）を一覧に反映
    } catch (e) { say("カメラを使えません。Safari の設定でカメラを許可してください（" + (e.name || e) + "）"); }
  }
  if (live.stream) { video.srcObject = live.stream; video.play().catch(() => { }); }

  /* 手順パネル */
  let mode = null; // "corners" | "colors"
  // 試合前の設定は映像の下に置き、済んだら折りたたむ（ライブ中は記録ボタンを映像の横に常に表示）
  const stepsBody = h("div", { class: "stack" });
  const ready = () => cfg.corners.length === 4 && ["own", "opp"].every(k => cfg.samples.some(s => s.cat === k));
  const steps = h("details", { class: "panel", open: !ready() }, h("summary", { class: "lv-sum" }, "試合前の設定（四隅・色・攻撃方向）"), stepsBody);
  left.append(steps);
  const showStage = () => { try { window.scrollTo(0, Math.max(0, stage.getBoundingClientRect().top + window.scrollY - 70)); } catch { } };
  function drawSteps() {
    const c = cfg.corners.length === 4, cs = ["own", "opp"].every(k => cfg.samples.some(s => s.cat === k));
    steps.querySelector("summary").textContent = "試合前の設定（四隅・色・攻撃方向）" + (c && cs ? " ✓ 設定済み" : " ⚠ 未設定");
    stepsBody.replaceChildren(
      h("div", null, h("b", null, "① ピッチの四隅"), " ", c ? h("span", { class: "pill ok" }, "設定済み") : h("span", { class: "pill warn" }, "未設定")),
      h("div", { class: "small muted" }, "映像の上で 左奥 → 右奥 → 右手前 → 左手前 の順にタップ。"),
      h("button", { class: "btn sm" + (mode === "corners" ? " primary" : ""), onclick: () => { mode = "corners"; cfg.corners = []; saveCfg(); drawSteps(); drawOverlay(); showStage(); } }, mode === "corners" ? `四隅をタップ中（${cfg.corners.length}/4）` : "四隅を設定する"),
      h("div", null, h("b", null, "② ユニフォームの色"), " ", cs ? h("span", { class: "pill ok" }, `登録 ${cfg.samples.length}人`) : h("span", { class: "pill warn" }, "未登録")),
      h("div", { class: "small muted" }, "「色を登録」を押すと人に枠が出ます。枠をタップして、自チーム・相手・除外（審判・PB・GK）を各1人以上登録。"),
      h("div", { class: "row" },
        h("button", { class: "btn sm" + (mode === "colors" ? " primary" : ""), onclick: async () => { mode = "colors"; drawSteps(); showStage(); await detectOnce(); } }, mode === "colors" ? "枠をタップして登録中" : "色を登録"),
        cfg.samples.length ? h("button", { class: "btn sm ghost", onclick: () => { cfg.samples = []; saveCfg(); drawSteps(); drawOverlay(); } }, "色をリセット") : null),
      h("div", null, h("b", null, "③ 自チームの攻撃方向（前半）")),
      h("div", { class: "seg" }, [["left", "← 左のゴール"], ["right", "右のゴール →"]].map(([v, l]) => h("button", { class: cfg.ownAttack === v ? "on" : "", onclick: () => { cfg.ownAttack = v; saveCfg(); drawSteps(); } }, l))),
      h("div", { class: "small muted" }, "後半（Q3・Q4）は自動で逆向きにします。"));
  }

  /* クォーター操作と手入力 */
  const ctl = h("div", { class: "panel stack" });
  side.append(ctl);
  function drawCtl() {
    const q = live.q;
    const tbtn = (team, label, fn) => h("button", { class: "evbtn" + (team === "opp" ? " opp" : ""), disabled: !q.running, onclick: fn }, label);
    const add = (team, key, ev, res) => async () => { q.counts[team][key]++; await logTag(team, ev, res, false); toast(`${team === "own" ? ownName : oppName}: ${key}`); drawCtl(); drawBoard(); };
    ctl.replaceChildren(
      h("div", { class: "row spread" },
        h("div", { class: "seg" }, [1, 2, 3, 4].map(n => h("button", { class: q.n === n ? "on" : "", disabled: q.running, onclick: () => { live.q = newQuarter(n); live.q.matchId = m.id; drawCtl(); drawBoard(); } }, "Q" + n))),
        h("span", { class: "timecode", id: "lv_clock" }, fmt(qTime()).slice(0, -2))),
      q.running
        ? h("button", { class: "btn danger", onclick: async () => { tickPossession(); q.elapsed = qTime(); q.running = false; await saveQuarter(); drawCtl(); showSummary(m); } }, `Q${q.n} 終了 → 集計を表示`)
        : h("button", { class: "btn primary", onclick: () => {
          if (!live.sess) return toast("先にカメラ（または動画）を開始し、設定を済ませてください");
          q.running = true; q.t0 = performance.now(); q.lastTick = q.t0; drawCtl(); runLoop();
        } }, `Q${q.n} 開始（AI 解析スタート）`),
      h("span", { class: "label" }, "ボール保持（入れ替わったらタップ）"),
      h("div", { class: "seg" },
        h("button", { class: q.holder === "own" ? "on" : "", onclick: () => { tickPossession(); q.holder = "own"; drawCtl(); } }, ownName),
        h("button", { class: q.holder === "opp" ? "on" : "", onclick: () => { tickPossession(); q.holder = "opp"; drawCtl(); } }, oppName),
        h("button", { class: !q.holder ? "on" : "", onclick: () => { tickPossession(); q.holder = null; drawCtl(); } }, "止まっている")),
      ...[["own", ownName], ["opp", oppName]].map(([t, name]) => h("div", { class: "stack", style: "gap:6px" },
        h("span", { class: "label" }, name),
        h("div", { class: "events", style: "grid-template-columns:repeat(3,1fr)" },
          tbtn(t, "シュート枠内", add(t, "シュート枠内", "シュート", "枠内(セーブ)")),
          tbtn(t, "シュート枠外", add(t, "シュート枠外", "シュート", "枠外")),
          tbtn(t, "ゴール", add(t, "ゴール", "シュート", "ゴール")),
          tbtn(t, "PC", add(t, "PC", "PC", null)),
          tbtn(t, "PS", add(t, "PS", "PS", null))))));
  }

  /* その場の数字 */
  const board = h("div", { class: "panel" });
  left.append(board);
  function drawBoard() { board.replaceChildren(summaryTable(live.q, ownName, oppName, true)); }

  /* オーバーレイ描画 */
  function drawOverlay() {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw) return;
    ov.width = vw; ov.height = vh;
    const g = ov.getContext("2d"); g.clearRect(0, 0, vw, vh);
    const lw = Math.max(2, vw / 640);
    const c = cfg.corners;
    if (c.length) {
      g.strokeStyle = "#f2c230"; g.lineWidth = lw * 1.5; g.beginPath();
      c.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]));
      if (c.length === 4) g.closePath(); g.stroke();
      c.forEach((p, i) => { g.fillStyle = "#f2c230"; g.beginPath(); g.arc(p[0], p[1], lw * 5, 0, 7); g.fill(); g.fillStyle = "#000"; g.font = `${lw * 9}px sans-serif`; g.fillText(String(i + 1), p[0] - lw * 3, p[1] + lw * 3); });
      const Hi = invHomography();
      if (Hi) { // 23m ラインとサークルを描いて、合わせ具合を確認できるように
        g.strokeStyle = "rgba(242,194,48,.9)"; g.lineWidth = lw;
        const line = pts => { g.beginPath(); pts.forEach(([x, y], i) => { const [u, v] = applyH(Hi, x, y); i ? g.lineTo(u, v) : g.moveTo(u, v); }); g.stroke(); };
        for (const x of [PITCH.D23, PITCH.L - PITCH.D23, PITCH.L / 2]) line([[x, 0], [x, PITCH.W]]);
        for (const gx of [0, PITCH.L]) {
          const pts = [], dir = gx === 0 ? 1 : -1, cy = PITCH.W / 2;
          for (let a = -90; a <= 0; a += 6) pts.push([gx + dir * PITCH.R * Math.cos(a * Math.PI / 180), cy - PITCH.POST + PITCH.R * Math.sin(a * Math.PI / 180)]);
          for (let a = 0; a <= 90; a += 6) pts.push([gx + dir * PITCH.R * Math.cos(a * Math.PI / 180), cy + PITCH.POST + PITCH.R * Math.sin(a * Math.PI / 180)]);
          line(pts);
        }
      }
    }
    const col = { own: "#3b82f6", opp: "#e7854f", ex: "#9aa5b1" };
    for (const d of live.dets) {
      if (!d.onPitch) continue;
      g.strokeStyle = col[d.cat] || "#ffffff"; g.lineWidth = lw * (d.cat ? 1.6 : 1);
      g.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    }
  }
  function toVideoXY(e) {
    const r = ov.getBoundingClientRect(), vw = video.videoWidth, vh = video.videoHeight;
    // object-fit: contain の余白を考慮
    const s = Math.min(r.width / vw, r.height / vh), dw = vw * s, dh = vh * s, ox = (r.width - dw) / 2, oy = (r.height - dh) / 2;
    return [(e.clientX - r.left - ox) / s, (e.clientY - r.top - oy) / s];
  }
  ov.addEventListener("pointerdown", async e => {
    if (!video.videoWidth) return;
    const [x, y] = toVideoXY(e);
    if (mode === "corners") {
      cfg.corners.push([x, y]); saveCfg();
      if (cfg.corners.length === 4) { mode = null; toast("四隅を設定しました。黄色の線がピッチのラインに重なるか確認してください"); }
      drawSteps(); drawOverlay();
    } else if (mode === "colors") {
      const d = live.dets.find(d => x >= d.x1 && x <= d.x2 && y >= d.y1 && y <= d.y2);
      if (!d || !d.lab) return toast("人の枠の中をタップしてください");
      const cat = await dialog("この人は？", h("p", null, "ユニフォームの色を登録します。審判・PB・GK・ベンチの人は「除外」にしてください。"),
        [{ label: "キャンセル", value: null }, { label: "除外", value: "ex" }, { label: oppName, value: "opp" }, { label: ownName, cls: "primary", value: "own" }]);
      if (!cat) return;
      cfg.samples.push({ cat, lab: d.lab }); saveCfg();
      annotate(live.dets); drawSteps(); drawOverlay();
      toast(`${CAT[cat]}として登録しました`);
    }
  });

  async function grab() {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw) return null;
    if (full.width !== vw) { full.width = vw; full.height = vh; }
    fullx.drawImage(video, 0, 0, vw, vh);
    return [vw, vh];
  }
  async function detectOnce() {
    if (!video.videoWidth) return toast("先にカメラ（または動画）を開始してください");
    try {
      await loadModel(say);
      const sz = await grab(); if (!sz) return;
      live.dets = annotate(await detect(full, sz[0], sz[1]));
      say(`AI: ${live.backend === "webgpu" ? "GPU" : "CPU"} で動作 ・ 1回 ${live.ms}ms ・ ${live.dets.filter(d => d.onPitch).length}人を検出`);
      drawOverlay();
    } catch (e) { say("AI を起動できませんでした: " + (e.message || e)); }
  }
  async function runLoop() {
    const id = ++live.loopId;
    try { await loadModel(say); } catch (e) { say("AI を起動できませんでした: " + (e.message || e)); return; }
    live.lastT = 0;
    while (id === live.loopId && live.q?.running && document.body.contains(video)) {
      const sz = await grab();
      if (sz) {
        live.dets = annotate(await detect(full, sz[0], sz[1]));
        updateZones(live.dets, qTime());
        tickPossession();
        drawOverlay(); drawBoard();
        say(`解析中 ・ ${live.backend === "webgpu" ? "GPU" : "CPU"} ・ 1回 ${live.ms}ms ・ 検出 ${live.dets.filter(d => d.onPitch).length}人`);
      }
      const clk = document.getElementById("lv_clock"); if (clk) clk.textContent = fmt(qTime()).slice(0, -2);
      await new Promise(r => setTimeout(r, 30));
    }
  }

  video.addEventListener("loadedmetadata", drawOverlay);
  window.__viewCleanup = () => { live.loopId++; if (live.q) { tickPossession(); if (live.q.running) { live.q.elapsed = qTime(); live.q.running = false; } } };
  drawSteps(); drawCtl(); drawBoard();
  if (live.stream || video.src) drawOverlay();
  return wrap;
};

function stopLive() {
  live.loopId++;
  if (live.stream) { live.stream.getTracks().forEach(t => t.stop()); live.stream = null; }
}

/* ---------- 集計表（Q間の休憩で見せる） */
function pct(a, b) { return a + b > 0 ? Math.round(a / (a + b) * 100) : null; }
function summaryTable(q, ownName, oppName, compact) {
  if (!q) return h("div");
  const p = pct(q.poss.own, q.poss.opp), t = pct(q.terr.own, q.terr.opp);
  const C = q.counts;
  const rows = [
    ["ボール保持率", p == null ? "—" : p + "%", p == null ? "—" : (100 - p) + "%", "タップ"],
    ["相手陣でプレー", t == null ? "—" : t + "%", t == null ? "—" : (100 - t) + "%", "AI"],
    ["23m侵入", C.own["23m侵入"], C.opp["23m侵入"], "AI"],
    ["サークル侵入", C.own["サークル侵入"], C.opp["サークル侵入"], "AI"],
    ["シュート（合計）", C.own["シュート枠内"] + C.own["シュート枠外"] + C.own["ゴール"], C.opp["シュート枠内"] + C.opp["シュート枠外"] + C.opp["ゴール"], "タップ"],
    ["　枠内（ゴール含む）", C.own["シュート枠内"] + C.own["ゴール"], C.opp["シュート枠内"] + C.opp["ゴール"], "タップ"],
    ["　枠外", C.own["シュート枠外"], C.opp["シュート枠外"], "タップ"],
    ["ゴール", C.own["ゴール"], C.opp["ゴール"], "タップ"],
    ["PC", C.own.PC, C.opp.PC, "タップ"],
    ["PS", C.own.PS, C.opp.PS, "タップ"],
  ];
  return h("div", { class: "tablewrap" }, h("table", { class: compact ? "" : "big" },
    h("thead", null, h("tr", null, h("th", null, `Q${q.n}`), h("th", { class: "n" }, ownName), h("th", { class: "n" }, oppName), h("th", null, "記録"))),
    h("tbody", null, rows.map(r => h("tr", null, h("td", null, r[0]), h("td", { class: "n" }, String(r[1])), h("td", { class: "n" }, String(r[2])), h("td", null, h("span", { class: "pill" + (r[3] === "AI" ? " ai" : "") }, r[3])))))));
}
async function showSummary(m) {
  const ownName = team(m.teamId)?.name ?? "自チーム", oppName = team(m.oppId)?.name ?? "相手";
  const rec = (await DB.get("meta", "live:" + m.id)) || { quarters: {} };
  // 試合通算
  const tot = newQuarter(0); tot.n = "計";
  for (const k of Object.keys(rec.quarters)) {
    const r = rec.quarters[k];
    for (const s of ["own", "opp"]) { tot.poss[s] += r.poss[s]; tot.terr[s] += r.terr[s]; for (const c in r.counts[s]) tot.counts[s][c] += r.counts[s][c]; }
  }
  await dialog(`Q${live.q.n} の集計`, h("div", { class: "stack" },
    summaryTable(live.q, ownName, oppName, false),
    Object.keys(rec.quarters).length > 1 ? h("h3", null, "試合通算") : null,
    Object.keys(rec.quarters).length > 1 ? summaryTable(tot, ownName, oppName, false) : null,
    h("p", { class: "small muted", style: "margin:0" }, "「AI」の数字は推定です。試合後に動画で確認してください。")),
    [{ label: "閉じる", cls: "primary", value: true }]);
}
