#!/usr/bin/env python3
"""TCP server for remote training task creation.

Typical usage:
  python3 server.py --host 0.0.0.0 --port 9884
"""

from __future__ import annotations

import argparse
import json
import selectors
import socket
import sys
from dataclasses import dataclass
from typing import Any


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9884
DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_RECV_BYTES = 4096
TASKS_BY_USER: dict[str, list[dict[str, str]]] = {}


@dataclass
class Client:
    address: tuple[str, int]
    socket: socket.socket


def log(message: str) -> None:
    print(f"[tcp_read_server] {message}", flush=True)


def make_server_socket(host: str, port: int, backlog: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(backlog)
    server.setblocking(False)
    return server


def close_client(selector: selectors.BaseSelector, client: Client, reason: str) -> None:
    try:
        selector.unregister(client.socket)
    except (KeyError, ValueError):
        pass
    try:
        client.socket.close()
    finally:
        log(f"closed {format_address(client.address)}: {reason}")


def format_address(address: tuple[str, int]) -> str:
    host, port = address
    return f"{host}:{port}"


def accept_clients(
    selector: selectors.BaseSelector,
    server: socket.socket,
    max_connections: int,
) -> None:
    while True:
        try:
            client_socket, address = server.accept()
        except BlockingIOError:
            return

        active_connections = len(selector.get_map()) - 1
        if active_connections >= max_connections:
            log(f"rejecting {format_address(address)}: max connections reached")
            client_socket.close()
            continue

        client_socket.setblocking(False)
        client = Client(address=address, socket=client_socket)
        selector.register(client_socket, selectors.EVENT_READ, data=client)
        log(f"accepted {format_address(address)} ({active_connections + 1}/{max_connections})")


def read_client(
    selector: selectors.BaseSelector,
    client: Client,
    recv_bytes: int,
    encoding: str,
) -> None:
    try:
        data = client.socket.recv(recv_bytes)
    except ConnectionResetError:
        close_client(selector, client, "connection reset")
        return
    except OSError as exc:
        close_client(selector, client, f"read error: {exc}")
        return

    if not data:
        close_client(selector, client, "peer closed")
        return

    text = data.decode(encoding, errors="replace")
    response = handle_request(text)
    client.socket.sendall(json.dumps(response, ensure_ascii=False).encode(encoding))
    close_client(selector, client, "response sent")


def handle_request(text: str) -> dict[str, Any]:
    try:
        request = json.loads(text)
    except json.JSONDecodeError:
        return {"message": "invalid json", "tasks": []}

    username = str(request.get("username") or "").strip()
    task_name = str(request.get("taskName") or "").strip()
    action = str(request.get("action") or "").strip()
    if not username or not task_name or action != "开始训练":
        return {"message": "invalid request", "tasks": TASKS_BY_USER.get(username, [])}

    tasks = TASKS_BY_USER.setdefault(username, [])
    if any(task["taskName"] == task_name for task in tasks):
        message = "create task failed"
    else:
        tasks.append({"taskName": task_name, "status": ""})
        message = "create task success"

    return {"message": message, "tasks": tasks}


def serve(args: argparse.Namespace) -> None:
    selector = selectors.DefaultSelector()
    server = make_server_socket(args.host, args.port, args.max_connections)
    selector.register(server, selectors.EVENT_READ, data=None)

    log(
        f"listening on {args.host}:{args.port}, "
        f"max_connections={args.max_connections}, recv_bytes={args.recv_bytes}"
    )

    try:
        while True:
            for key, _ in selector.select():
                if key.data is None:
                    accept_clients(selector, server, args.max_connections)
                else:
                    read_client(selector, key.data, args.recv_bytes, args.encoding)
    except KeyboardInterrupt:
        log("stopping")
    finally:
        for key in list(selector.get_map().values()):
            sock = key.fileobj
            try:
                selector.unregister(sock)
            except (KeyError, ValueError):
                pass
            sock.close()
        selector.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maintain many TCP connections and print received data.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host, default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Bind port, default: {DEFAULT_PORT}")
    parser.add_argument(
        "--max-connections",
        type=int,
        default=DEFAULT_MAX_CONNECTIONS,
        help=f"Maximum active client sockets, default: {DEFAULT_MAX_CONNECTIONS}",
    )
    parser.add_argument(
        "--recv-bytes",
        type=int,
        default=DEFAULT_RECV_BYTES,
        help=f"Bytes to read per socket event, default: {DEFAULT_RECV_BYTES}",
    )
    parser.add_argument("--encoding", default="utf-8", help="Text encoding for terminal output, default: utf-8")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_connections < 1:
        print("--max-connections must be at least 1", file=sys.stderr)
        return 2
    if args.recv_bytes < 1:
        print("--recv-bytes must be at least 1", file=sys.stderr)
        return 2

    serve(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
