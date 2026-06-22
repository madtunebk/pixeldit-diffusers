"""
Download images from Pexels by search query.
Get a free API key instantly at https://www.pexels.com/api/

Usage:
    python scripts/download_unsplash.py --key YOUR_KEY --query "yarn art" --n 50 --out /home/nobus/HDD/lora_yarn2
"""

import argparse
import os
import requests
from pathlib import Path
from tqdm import tqdm

def main():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--key",   default=os.getenv("pexels"), help="Pexels API key")
    ap.add_argument("--query", required=True, help="search query")
    ap.add_argument("--n",     type=int, default=50)
    ap.add_argument("--out",   required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    headers = {"Authorization": args.key}
    saved = 0
    page  = 1

    with tqdm(total=args.n, desc="downloading") as bar:
        while saved < args.n:
            r = requests.get("https://api.pexels.com/v1/search",
                headers=headers,
                params={"query": args.query, "per_page": 30, "page": page}
            )
            results = r.json().get("photos", [])
            if not results:
                break

            for photo in results:
                if saved >= args.n:
                    break
                url  = photo["src"]["large"]
                desc = photo.get("alt") or args.query
                desc = desc.strip().replace("\n", " ")

                img_data = requests.get(url, timeout=15).content
                stem = f"{saved:05d}"
                open(os.path.join(args.out, f"{stem}.jpg"), "wb").write(img_data)
                Path(os.path.join(args.out, f"{stem}.txt")).write_text(desc)
                saved += 1
                bar.update(1)
            page += 1

    print(f"Done. {saved} images saved to {args.out}/")

if __name__ == "__main__":
    main()
