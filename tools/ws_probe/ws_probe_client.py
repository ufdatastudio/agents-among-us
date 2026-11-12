"""Checks a WebSocket server from the outside, through whatever proxy is in front of it.

Usage:
    python ws_probe_client.py https://agents-among-us.rc.ufl.edu [--idle 75] [--no-quit]
"""

import argparse
import base64
import json
import os
import socket
import ssl
import struct
import threading
import time
import urllib.request
from urllib.parse import urlparse


def connect(url: str, path: str, origin: str | None = None) -> tuple[socket.socket, str, bytes]:
    """Opens a connection, sends a WebSocket handshake and returns the socket, the status line and the response head."""
    parts = urlparse(url)
    secure = parts.scheme in ("https", "wss")
    port = parts.port or (443 if secure else 80)
    raw = socket.create_connection((parts.hostname, port), timeout=15)
    sock = ssl.create_default_context().wrap_socket(raw, server_hostname=parts.hostname) if secure else raw
    key = base64.b64encode(os.urandom(16)).decode()
    lines = [f"GET {path} HTTP/1.1", f"Host: {parts.netloc}", "Upgrade: websocket", "Connection: Upgrade", f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"]
    if origin:
        lines.append(f"Origin: {origin}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sock.recv(4096)
        if not chunk:
            break
        head += chunk
    status = head.split(b"\r\n", 1)[0].decode("latin-1")
    return sock, status, head


def read_exact(sock: socket.socket, n: int) -> bytes:
    """Reads exactly n bytes."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("closed by the peer")
        data += chunk
    return data


def send(sock: socket.socket, opcode: int, data: bytes = b"") -> None:
    """Sends one masked frame."""
    mask = os.urandom(4)
    n = len(data)
    head = bytes([0x80 | opcode])
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 65536:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack(">Q", n)
    masked = (int.from_bytes(data, "big") ^ int.from_bytes((mask * (n // 4 + 1))[:n], "big")).to_bytes(n, "big") if n else b""
    sock.sendall(head + mask + masked)


def receive(sock: socket.socket, timeout: float) -> tuple[int, bytes]:
    """Reads one frame of the server, which is unmasked."""
    sock.settimeout(timeout)
    b1, b2 = read_exact(sock, 2)
    n = b2 & 0x7F
    if n == 126:
        n = struct.unpack(">H", read_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", read_exact(sock, 8))[0]
    return b1 & 0x0F, read_exact(sock, n)


def open_ws(url: str, origin: str | None = None):
    """Opens a WebSocket on /ws, or raises with what came back instead of the 101."""
    sock, status, head = connect(url, "/ws", origin)
    if " 101 " not in status + " ":
        raise RuntimeError(f"handshake answered '{status}'. Head and body: {head.decode('latin-1')[:900]}")
    return sock


def report(name: str, ok: bool, detail: str = "") -> bool:
    """Prints one result line."""
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def main() -> None:
    """Runs the checks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--idle", type=float, default=75)
    parser.add_argument("--no-quit", action="store_true")
    args = parser.parse_args()
    results = []
    try:
        with urllib.request.urlopen(args.url + "/headers", timeout=20) as r:
            print("headers the server saw on a plain GET:", json.dumps(json.loads(r.read()))[:600], flush=True)
    except Exception as e:  # noqa: BLE001
        results.append(report("plain GET /headers", False, str(e)))
        print("The probe is not answering. Stopping.")
        return
    try:
        sock = open_ws(args.url, origin=args.url)
    except Exception as e:  # noqa: BLE001
        report("handshake", False, str(e))
        return
    results.append(report("handshake (101 Switching Protocols)", True))
    send(sock, 1, b"hello")
    results.append(report("text echo", receive(sock, 10) == (1, b"hello")))
    send(sock, 2, os.urandom(1000))
    op, data = receive(sock, 10)
    results.append(report("binary 1 KB echo", op == 2 and len(data) == 1000))
    big = os.urandom(200_000)
    send(sock, 2, big)
    op, data = receive(sock, 20)
    results.append(report("binary 200 KB echo", op == 2 and data == big))
    for i in range(20):
        send(sock, 1, f"burst {i}".encode())
    got = [receive(sock, 10)[1].decode() for _ in range(20)]
    results.append(report("20 messages in a burst, in order", got == [f"burst {i}" for i in range(20)]))
    send(sock, 9, b"ping")
    results.append(report("ping answered with pong", receive(sock, 10) == (10, b"ping")))

    outcome = {}

    def silent() -> None:
        try:
            s = open_ws(args.url)
            time.sleep(args.idle)
            send(s, 1, b"after silence")
            outcome["silent"] = receive(s, 15) == (1, b"after silence")
            s.close()
        except Exception as e:  # noqa: BLE001
            outcome["silent"] = f"{type(e).__name__}: {e}"

    def ticks() -> None:
        try:
            s = open_ws(args.url)
            send(s, 1, b"ticks:3:20")
            outcome["ticks"] = [receive(s, 30)[1].decode() for _ in range(3)]
            s.close()
        except Exception as e:  # noqa: BLE001
            outcome["ticks"] = f"{type(e).__name__}: {e}"

    print(f"running the idle checks, {args.idle:.0f} s ...", flush=True)
    threads = [threading.Thread(target=silent), threading.Thread(target=ticks)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    results.append(report(f"connection silent for {args.idle:.0f} s still works", outcome.get("silent") is True, "" if outcome.get("silent") is True else str(outcome.get("silent"))))
    results.append(report("server messages 20 s apart arrive", isinstance(outcome.get("ticks"), list), str(outcome.get("ticks"))))
    send(sock, 8, struct.pack(">H", 1000))
    try:
        op, _ = receive(sock, 10)
        results.append(report("close handshake", op == 8))
    except Exception as e:  # noqa: BLE001
        results.append(report("close handshake", False, str(e)))
    print(f"\n{sum(results)} of {len(results)} checks passed")
    if not args.no_quit:
        try:
            urllib.request.urlopen(args.url + "/quit", timeout=20).read()
            print("asked the probe to stop")
        except Exception as e:  # noqa: BLE001
            print("could not ask the probe to stop:", e)


if __name__ == "__main__":
    main()
