"""Phase 2 §12: download SQID product photos (test + supplementary) -> data/images/<product_id>.jpg.

Resumable (skips files already on disk); failures logged to runs/p2/image_failures.csv.
  python scripts/p2_download_images.py [--limit N]
"""
import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmsearch.esci import ROOT, RUNS, image_urls  # noqa: E402

IMAGES = ROOT / "data" / "images"
CONCURRENCY = 48


async def fetch(session, sem, pid, url, failures):
    dst = IMAGES / f"{pid}.jpg"
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status == 200:
                        body = await r.read()
                        tmp = dst.with_suffix(".part")
                        tmp.write_bytes(body)
                        tmp.rename(dst)
                        return
                    if r.status in (403, 404):
                        failures.append((pid, url, r.status))
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(2 ** attempt)
        failures.append((pid, url, "error"))


async def main_async(todo):
    failures = []
    sem = asyncio.Semaphore(CONCURRENCY)
    headers = {"User-Agent": "Mozilla/5.0 (research image fetch)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [fetch(session, sem, pid, url, failures) for pid, url in todo]
        for i in range(0, len(tasks), 2000):
            await asyncio.gather(*tasks[i:i + 2000])
            print(f"{min(i + 2000, len(tasks)):,}/{len(tasks):,} done, {len(failures):,} failed", flush=True)
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    urls = image_urls()
    have = {p.stem for p in IMAGES.glob("*.jpg")}
    todo = [(p, u) for p, u in zip(urls["product_id"], urls["image_url"]) if p not in have]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(urls):,} products with a URL, {len(have):,} on disk, {len(todo):,} to fetch", flush=True)
    failures = asyncio.run(main_async(todo))
    out = RUNS / "p2"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["product_id", "image_url", "status"]).to_csv(out / "image_failures.csv", index=False)
    n = len(list(IMAGES.glob("*.jpg")))
    print(f"on disk: {n:,} / {len(urls):,} ({100 * n / len(urls):.1f}%)")


if __name__ == "__main__":
    main()
