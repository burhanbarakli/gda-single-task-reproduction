#!/usr/bin/env python3
"""Serve the report with a small live JSON endpoint; no third-party web package."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from build_report import ROOT, collect, render_html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    source = args.source_root.expanduser().resolve()
    assets = (ROOT / "reports/assets").resolve()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, mime: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self.send_bytes(render_html(collect(source)).encode(), "text/html; charset=utf-8")
                return
            if path == "/api/data":
                self.send_bytes(json.dumps(collect(source), ensure_ascii=False).encode(), "application/json; charset=utf-8")
                return
            if path.startswith("/assets/"):
                requested = (ROOT / "reports" / unquote(path.lstrip("/"))).resolve()
                if assets == requested or assets in requested.parents:
                    if requested.is_file():
                        self.send_bytes(requested.read_bytes(), mimetypes.guess_type(requested.name)[0] or "application/octet-stream")
                        return
            self.send_bytes(b"not found", "text/plain", 404)

        def log_message(self, fmt, *values):
            print(fmt % values)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"GDA dashboard: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

