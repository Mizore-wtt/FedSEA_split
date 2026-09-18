"""Authenticated, certificate-verified loopback HTTPS. No redirects or retries."""

import hmac
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import socket
from socketserver import ThreadingMixIn
import ssl
import threading
import time
from urllib.parse import urlsplit

from .protocol import CONTENT_TYPE, pack, unpack

ROUTES = {"/v1/health", "/v1/session", "/v1/forward", "/v1/crop", "/v1/close"}


def read_token(directory):
    token = (Path(directory) / "token.txt").read_text(encoding="ascii").strip()
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("Invalid local bearer token; prepare credentials again.")
    return token


class RpcError(RuntimeError):
    pass


class HttpsRpc:
    def __init__(self, url, directory, timeout=60, limit=16 * 1024**2):
        address = urlsplit(url)
        if (address.scheme != "https" or address.hostname not in ("localhost", "127.0.0.1")
                or address.username or address.password or address.query or address.fragment
                or address.path not in ("", "/") or address.port is None):
            raise ValueError("Only explicit HTTPS loopback endpoints are allowed.")
        self.host, self.port = address.hostname, address.port
        self.context = ssl.create_default_context(cafile=str(Path(directory) / "ca.crt"))
        self.context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.token, self.timeout, self.limit = read_token(directory), timeout, limit
        self.stats = {"requests": 0, "sent_body_bytes": 0, "received_body_bytes": 0, "rpc_seconds": 0.0}

    def call(self, path, metadata, tensors=None):
        # Forward mutates remote KV. A lost reply must not cause the same tokens to be appended twice.
        if path not in ROUTES:
            raise ValueError("Unknown RPC route.")
        body = pack(metadata, tensors, self.limit)
        connection = http.client.HTTPSConnection(
            self.host, self.port, timeout=self.timeout, context=self.context
        )
        start = time.perf_counter()
        self.stats["requests"] += 1
        try:
            connection.request("POST", path, body=body, headers={
                "Authorization": "Bearer " + self.token, "Content-Type": CONTENT_TYPE,
                "Connection": "close",
            })
            self.stats["sent_body_bytes"] += len(body)
            response = connection.getresponse()
            sizes = response.headers.get_all("Content-Length", [])
            if (len(sizes) != 1 or not sizes[0].isascii() or not sizes[0].isdigit()
                    or not 4 <= int(sizes[0]) <= self.limit
                    or response.getheader("Transfer-Encoding") is not None
                    or response.getheader("Content-Encoding") is not None
                    or response.getheader("Content-Type") != CONTENT_TYPE):
                raise RpcError("Invalid HTTPS response framing.")
            data = response.read(int(sizes[0]) + 1)
            if len(data) != int(sizes[0]):
                raise RpcError("Truncated HTTPS response; session must reset.")
            self.stats["received_body_bytes"] += len(data)
            meta, result = unpack(data, self.limit)
            if response.status != 200:
                raise RpcError(f"HTTPS {response.status}: {meta.get('error', 'request rejected')}")
            return meta, result
        except (OSError, http.client.HTTPException) as error:
            raise RpcError(f"HTTPS connection failed ({type(error).__name__}); no forward was retried.") from error
        finally:
            connection.close()
            self.stats["rpc_seconds"] += time.perf_counter() - start


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def setup(self):
        self.request = self.server.tls.wrap_socket(self.request, server_side=True)
        super().setup()

    def finish(self):
        try:
            super().finish()
        finally:
            # wrap_socket detached the accepted raw socket; close the actual TLS owner.
            try:
                self.connection.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            self.connection.close()

    def log_message(self, format, *args):
        pass

    def reply(self, status, meta, tensors=None):
        body = pack(meta, tensors, self.server.state.net["max_body_bytes"])
        self.send_response(status)
        self.send_header("Content-Type", CONTENT_TYPE)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_POST(self):
        meta = {}
        try:
            auth = self.headers.get_all("Authorization", [])
            if len(auth) != 1 or not hmac.compare_digest(
                auth[0].encode("utf-8"), ("Bearer " + self.server.token).encode("ascii")
            ):
                self.reply(401, {"error": "Authentication required."})
                return
            if self.path not in ROUTES:
                self.reply(404, {"error": "Unknown route."})
                return
            sizes = self.headers.get_all("Content-Length", [])
            if (len(sizes) != 1 or not sizes[0].isascii() or not sizes[0].isdigit()
                    or self.headers.get("Transfer-Encoding") is not None
                    or self.headers.get("Content-Encoding") is not None
                    or self.headers.get("Expect") is not None
                    or self.headers.get_all("Content-Type") != [CONTENT_TYPE]):
                self.reply(400, {"error": "Invalid request framing."})
                return
            size = int(sizes[0])
            limit = self.server.state.net["max_body_bytes"]
            if not 4 <= size <= limit:
                self.reply(413, {"error": "Request body exceeds limits."})
                return
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("Truncated request.")
            meta, tensors = unpack(body, limit)
            output, data = self.server.state.dispatch(self.path, meta, tensors)
            self.reply(200, output, data)
        except (ValueError, KeyError, TypeError, OverflowError, RuntimeError) as error:
            self.discard(meta)
            self.reply(400, {"error": str(error)[:200]})
        except (OSError, TimeoutError):
            self.discard(meta)
        except Exception:
            self.discard(meta)
            self.reply(500, {"error": "Server operation failed; reset the session."})

    def discard(self, meta):
        sid = meta.get("session")
        if isinstance(sid, str):
            with self.server.state.lock:
                self.server.state.sessions.pop(sid, None)


class LoopbackServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    request_queue_size = 8
    allow_reuse_address = False

    def __init__(self, state, directory, port):
        self.state, self.token = state, read_token(directory)
        self.tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.tls.minimum_version = ssl.TLSVersion.TLSv1_2
        self.tls.load_cert_chain(str(Path(directory) / "server.crt"), str(Path(directory) / "server.key"))
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(("127.0.0.1", port), Handler)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(self.state.net["timeout_seconds"])
        return request, address

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # Expected for failed TLS handshakes/disconnects; no credentials or input logs.
        pass

    def service_actions(self):
        self.state.expire()

    def server_close(self):
        super().server_close()
        with self.state.lock:
            self.state.sessions.clear()
