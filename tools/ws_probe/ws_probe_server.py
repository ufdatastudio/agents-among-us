"""A WebSocket echo server that uses only the standard library, to test whether a proxy passes WebSocket upgrades.

Paths:
    /          a test page that runs the checks from a browser
    /headers   the request headers the server received, as JSON
    /ws        a WebSocket echo. A text message "ticks:N:S" makes the server send N messages, S seconds apart.
    /quit      stops the server

The server stops by itself after --ttl seconds, so it cannot be left running by mistake. It uses print and the standard
library only, so it runs in any Python image without installing packages.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import struct
import time

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
STOP = asyncio.Event()
START = time.time()

PAGE = """<!doctype html>
<meta charset="utf-8"><title>WebSocket probe</title>
<body style="font-family: monospace"><h3>WebSocket probe</h3><pre id="out"></pre>
<script>
const out = document.getElementById('out');
const log = (m) => { out.textContent += m + '\\n'; };
const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
function open() {
  return new Promise((res, rej) => {
    const ws = new WebSocket(url); ws.binaryType = 'arraybuffer';
    ws.onopen = () => res(ws);
    ws.onerror = () => rej(new Error('error before open'));
  });
}
function next(ws, ms) {
  return new Promise((res, rej) => {
    const t = setTimeout(() => rej(new Error('timeout')), ms);
    ws.onmessage = (e) => { clearTimeout(t); res(e.data); };
    ws.onclose = () => { clearTimeout(t); rej(new Error('closed')); };
  });
}
async function run() {
  try {
    log('url: ' + url);
    const ws = await open(); log('open: PASS');
    ws.send('hello'); log('text echo: ' + ((await next(ws, 5000)) === 'hello' ? 'PASS' : 'FAIL'));
    ws.send(new Uint8Array(100000).fill(7)); const back = await next(ws, 5000);
    log('binary 100 KB echo: ' + (back.byteLength === 100000 ? 'PASS' : 'FAIL'));
    log('idle 75 s ...'); await new Promise((r) => setTimeout(r, 75000));
    ws.send('after idle'); log('echo after 75 s idle: ' + ((await next(ws, 5000)) === 'after idle' ? 'PASS' : 'FAIL'));
    ws.close(); log('done'); document.title = 'probe finished';
  } catch (e) { log('FAIL: ' + e.message); document.title = 'probe failed'; }
}
run();
</script>
"""


async def read_request(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str]]:
    """Reads the request line and the headers of an HTTP request."""
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, target, _ = lines[0].split(" ", 2)
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return method, target, headers


async def read_frame(reader: asyncio.StreamReader) -> tuple[bool, int, bytes]:
    """Reads one WebSocket frame and unmasks it."""
    b1, b2 = await reader.readexactly(2)
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack(">H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", await reader.readexactly(8))[0]
    mask = await reader.readexactly(4) if b2 & 0x80 else b""
    data = await reader.readexactly(length)
    if mask and length:
        key = (mask * (length // 4 + 1))[:length]
        data = (int.from_bytes(data, "big") ^ int.from_bytes(key, "big")).to_bytes(length, "big")
    return bool(b1 & 0x80), b1 & 0x0F, data


def frame(opcode: int, data: bytes = b"") -> bytes:
    """Builds an unmasked WebSocket frame, which is what a server sends."""
    n = len(data)
    if n < 126:
        head = bytes([0x80 | opcode, n])
    elif n < 65536:
        head = bytes([0x80 | opcode, 126]) + struct.pack(">H", n)
    else:
        head = bytes([0x80 | opcode, 127]) + struct.pack(">Q", n)
    return head + data


def respond(writer: asyncio.StreamWriter, status: str, body: str, content_type: str = "text/plain") -> None:
    """Writes a complete HTTP response and asks the peer to close."""
    raw = body.encode()
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Type: {content_type}; charset=utf-8\r\nContent-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode() + raw
    )


async def send_ticks(writer: asyncio.StreamWriter, count: int, seconds: float) -> None:
    """Sends `count` text messages, `seconds` apart."""
    for i in range(1, count + 1):
        await asyncio.sleep(seconds)
        writer.write(frame(1, f"tick {i} at {time.time() - START:.0f} s".encode()))
        await writer.drain()


async def websocket(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict[str, str], peer: str) -> None:
    """Completes the handshake and echoes messages until the peer closes."""
    accept = base64.b64encode(hashlib.sha1((headers["sec-websocket-key"] + GUID).encode()).digest()).decode()
    writer.write(f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n".encode())
    await writer.drain()
    opened = time.time()
    print(f"[{time.time() - START:6.1f} s] ws open from {peer}, origin={headers.get('origin')}", flush=True)
    tickers = []
    kind, message = 1, b""
    try:
        while True:
            fin, opcode, data = await read_frame(reader)
            if opcode == 8:
                writer.write(frame(8, data[:2]))
                break
            if opcode == 9:
                writer.write(frame(10, data))
            elif opcode in (1, 2):
                kind, message = opcode, data
            elif opcode == 0:
                message += data
            if fin and opcode in (0, 1, 2):
                if kind == 1 and message.startswith(b"ticks:"):
                    _, count, seconds = message.decode().split(":")
                    tickers.append(asyncio.create_task(send_ticks(writer, int(count), float(seconds))))
                else:
                    writer.write(frame(kind, message))
                await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError):
        pass
    finally:
        for task in tickers:
            task.cancel()
        print(f"[{time.time() - START:6.1f} s] ws closed from {peer} after {time.time() - opened:.0f} s", flush=True)


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Routes one connection."""
    peer = str(writer.get_extra_info("peername"))
    try:
        _, target, headers = await read_request(reader)
        path = target.split("?")[0]
        print(f"[{time.time() - START:6.1f} s] {target} from {peer} upgrade={headers.get('upgrade')} connection={headers.get('connection')}", flush=True)
        if path == "/ws" and headers.get("upgrade", "").lower() == "websocket" and "sec-websocket-key" in headers:
            await websocket(reader, writer, headers, peer)
        elif path == "/ws":
            respond(writer, "400 Bad Request", json.dumps({"problem": "no WebSocket upgrade reached the server", "headers_seen": headers}, indent=1), "application/json")
        elif path == "/headers":
            respond(writer, "200 OK", json.dumps(headers, indent=1), "application/json")
        elif path == "/quit":
            respond(writer, "200 OK", "stopping")
            STOP.set()
        elif path == "/":
            respond(writer, "200 OK", PAGE, "text/html")
        else:
            respond(writer, "404 Not Found", "not found")
        await writer.drain()
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError, ValueError):
        pass
    finally:
        writer.close()


async def main(port: int, ttl: float) -> None:
    """Serves until /quit is requested or the time to live runs out."""
    server = await asyncio.start_server(handle, "0.0.0.0", port)
    print(f"probe listening on {port}, stops after {ttl:.0f} s", flush=True)
    try:
        await asyncio.wait_for(STOP.wait(), timeout=ttl)
    except asyncio.TimeoutError:
        print("time to live reached", flush=True)
    server.close()
    await asyncio.sleep(0.5)
    print("probe stopped", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ttl", type=float, default=420)
    args = parser.parse_args()
    asyncio.run(main(args.port, args.ttl))
