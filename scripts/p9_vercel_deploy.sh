#!/usr/bin/env bash
# Bundle app/ + runs/p9/catalog.sqlite + web/dist into runs/p9/esci-mmsearch and deploy it to Vercel (prod).
# Static files: public/ (CDN). API: api/index.py (one Python function, FastAPI). See docs/demo-handoff.md.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/runs/p9/esci-mmsearch   # folder name = Vercel project name

[ -f "$ROOT/runs/p9/catalog.sqlite" ] || { echo "run scripts/p9_build_db.py first"; exit 1; }
(cd "$ROOT/web" && npm run build)

mkdir -p "$OUT"
find "$OUT" -mindepth 1 -maxdepth 1 ! -name .vercel -exec rm -rf {} +   # keep the project link
mkdir -p "$OUT/api" "$OUT/runs/p9"
cp -r "$ROOT/app" "$OUT/app"
find "$OUT/app" -name __pycache__ -prune -exec rm -rf {} +
gzip -6 -c "$ROOT/runs/p9/catalog.sqlite" > "$OUT/runs/p9/catalog.sqlite.gz"   # Hobby plan: 100 MB per file
cp -r "$ROOT/web/dist" "$OUT/public"
echo 'from app.main import app  # noqa: F401  (Vercel serves this ASGI app)' > "$OUT/api/index.py"
echo 'fastapi==0.139.0' > "$OUT/requirements.txt"
cat > "$OUT/vercel.json" <<'EOF'
{
  "framework": null,
  "rewrites": [
    { "source": "/api/(.*)", "destination": "/api/index" },
    { "source": "/(.*)", "destination": "/index.html" }
  ],
  "functions": { "api/index.py": { "excludeFiles": "{public/**,.vercel/**}", "maxDuration": 10 } }
}
EOF
du -sh "$OUT"
cd "$OUT" && npx --yes vercel@latest deploy --prod --yes
