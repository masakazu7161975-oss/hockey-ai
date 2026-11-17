// オフライン用キャッシュ（アプリ本体のみ。データ・動画はキャッシュしない）
const CACHE = "hockey-ai-58a029fc7a";
const FILES = ["./", "./index.html", "./live.js", "./manifest.webmanifest", "./icon-180.png", "./icon-192.png", "./icon-512.png"];
const BIG = "hockey-ai-models-v1"; // AI モデル等の大きなファイルは別キャッシュ（毎回ダウンロードしない）
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES))); self.skipWaiting(); });
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE && k !== BIG).map(k => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const font = url.hostname.endsWith("fonts.googleapis.com") || url.hostname.endsWith("fonts.gstatic.com");
  if (url.origin !== location.origin && !font) return;
  if (/[/](ort|models)[/]/.test(url.pathname)) { // キャッシュ優先
    e.respondWith(caches.open(BIG).then(c => c.match(req).then(hit => hit || fetch(req).then(res => { if (res.ok) c.put(req, res.clone()); return res; }))));
    return;
  }
  // ネット優先（更新を反映）→ 失敗時はキャッシュ（試合会場などオフライン時）
  e.respondWith(fetch(req).then(res => {
    if (res.ok || res.type === "opaque") { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
    return res;
  }).catch(() => caches.match(req, { ignoreSearch: true }).then(r => r || caches.match("./index.html"))));
});
