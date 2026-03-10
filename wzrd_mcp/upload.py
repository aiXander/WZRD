"""Upload local files to S3 and print the public URL.

Uses content-based hashing so repeated uploads of the same file are instant.

Usage:
    python -m wzrd_mcp.upload file1.jpg file2.mp4 ...

Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_BUCKET_NAME
environment variables (and optionally CLOUDFRONT_URL, AWS_REGION_NAME).
"""

from __future__ import annotations

import argparse
import sys

from .file_io import upload


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload files to S3 for WZRD MCP")
    parser.add_argument("files", nargs="+", help="Local file paths to upload")
    args = parser.parse_args()

    for path in args.files:
        try:
            url = upload(path)
            print(url)
        except Exception as e:
            print(f"ERROR ({path}): {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
