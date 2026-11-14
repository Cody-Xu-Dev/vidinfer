"""仅面向本机/SSH 隧道的轻量 HTTP 服务；不提供任意文件或执行接口。"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import re
import secrets
from urllib.parse import urlsplit

from .backend import WebBackend


def create_server(backend, host="127.0.0.1", port=7860):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(
            "Bind locally and use SSH forwarding; public unauthenticated serving is not supported"
        )
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, data):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            # 防止 DNS rebinding；浏览器跨站请求不能借本机连接触发 GPU 任务。
            host = self.headers.get("Host", "")
            return (
                re.fullmatch(r"(?:127[.]0[.]0[.]1|localhost)(?::[0-9]{1,5})?", host)
                is not None
            )

        def do_GET(self):
            if not self.allowed():
                self.respond(403, {"error": "Invalid Host"})
                return
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    html = (
                        (files("vidinfer.web") / "assets/index.html")
                        .read_text(encoding="utf-8")
                        .replace("__VIDINFER_TOKEN__", token)
                    )
                    body = html.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/info":
                    self.respond(200, backend.info())
                elif path == "/api/jobs":
                    self.respond(200, backend.recent())
                elif match := re.fullmatch(r"/api/jobs/([0-9a-f]{32})", path):
                    self.respond(200, backend.status(match[1]))
                elif match := re.fullmatch(
                    r"/api/jobs/([0-9a-f]{32})/media/(video\.mp4|audio\.wav)", path
                ):
                    self.send_media(backend.media(match[1], match[2]))
                else:
                    self.respond(404, {"error": "Not found"})
            except (ValueError, FileNotFoundError) as error:
                self.respond(404, {"error": str(error)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                self.respond(500, {"error": f"{type(error).__name__}: {error}"})

        def do_POST(self):
            if not self.allowed() or not secrets.compare_digest(
                self.headers.get("X-VidInfer-Token", ""), token
            ):
                self.respond(403, {"error": "Invalid local request token"})
                return
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.respond(403, {"error": "Cross-origin requests are not allowed"})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.respond(415, {"error": "Expected application/json"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32768:
                    raise ValueError("Request must be 1..32768 bytes")
                data = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                if path == "/api/jobs":
                    self.respond(202, {"job_id": backend.submit(data)})
                elif match := re.fullmatch(r"/api/jobs/([0-9a-f]{32})/cancel", path):
                    self.respond(202, backend.cancel(match[1]))
                else:
                    self.respond(404, {"error": "Not found"})
            except (ValueError, TypeError, FileNotFoundError) as error:
                self.respond(400, {"error": str(error)})
            except RuntimeError as error:
                self.respond(409, {"error": str(error)})
            except Exception as error:
                self.respond(500, {"error": f"{type(error).__name__}: {error}"})

        def send_media(self, path):
            size = path.stat().st_size
            start, end = 0, size - 1
            status = 200
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if not match or not any(match.groups()):
                    self.respond(416, {"error": "Invalid Range"})
                    return
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), size - 1) if match[2] else size - 1
                else:
                    start = max(0, size - int(match[2]))
                if start >= size or start > end:
                    self.respond(416, {"error": "Range outside file"})
                    return
                status = 206
            self.send_response(status)
            self.send_header(
                "Content-Type", "video/mp4" if path.suffix == ".mp4" else "audio/wav"
            )
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    data = stream.read(min(65536, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)

    return ThreadingHTTPServer((host, port), Handler)


def serve(args):
    backend = WebBackend(
        args.workspace,
        checkpoint=args.checkpoint,
        resident_config=args.resident_config,
        demo=args.demo,
        text_cache=args.text_cache,
    )
    server = create_server(backend, args.host, args.port)
    print(f"VidInfer UI: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        backend.close()
