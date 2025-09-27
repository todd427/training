#!/usr/bin/env python3
"""
upload_to_HF.py - Upload a folder to HuggingFace Hub.

Usage:
    python upload_to_HF.py --repo_id "username/my-model" \
                    --repo_type model \
                    --folder /path/to/folder \
                    --private
"""

import argparse
from huggingface_hub import HfApi, upload_folder

def main():
    parser = argparse.ArgumentParser(description="Upload a folder to HuggingFace Hub")
    parser.add_argument("--repo_id", required=True, help="Repo ID (e.g., username/my-model)")
    parser.add_argument("--repo_type", choices=["model", "dataset", "space"], default="model", help="Repo type")
    parser.add_argument("--folder", required=True, help="Local folder path to upload")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    api = HfApi()
    # Ensure repo exists (does nothing if already exists and exist_ok=True)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True
    )

    print(f"Uploading {args.folder} to {args.repo_id} ({args.repo_type})...")
    upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=args.folder,
    )
    print("✅ Upload complete!")

if __name__ == "__main__":
    main()
