"""Start ``flextool-assistant --mcp`` and check the connection: stdout carries only protocol messages.

    python mcp_smoke.py [--tools N] [--call TOOL [--arguments JSON]] [--expect TEXT ...] -- COMMAND...

Starts COMMAND, sends ``initialize``, ``notifications/initialized`` and ``tools/list`` (and ``tools/call`` of
TOOL with --call) as newline-delimited JSON-RPC, each after the reply to the one before, then closes stdin and
reads stdout to its end.  Fails (exit 1) when a
stdout line is not a JSON-RPC 2.0 message, a request gets no reply or an error, the tool list does not have N
tools (--tools), the tool call is an error, or an --expect TEXT is in neither the instructions nor the tool
call's text.  Standard library only, so it also runs where FlexTool itself cannot be imported.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time

TIMEOUT_S = 300


def _exchange(command: list[str], requests: list[dict]) -> tuple[list[bytes] | None, str]:
    """Send *requests* one at a time, each after the reply to the one before, and keep stdin open until the last
    reply: the server stops serving at end of input, also requests still in progress.  Returns every stdout line
    (``None`` on a timeout) and stderr."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines: list[bytes] = []
    replies: "queue.Queue[object]" = queue.Queue()
    errors: list[bytes] = []

    def read_stdout() -> None:
        for raw in process.stdout:
            lines.append(raw.rstrip(b"\r\n"))
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if isinstance(message, dict) and "id" in message:
                replies.put(message["id"])
        replies.put(None)                       # end of output

    readers = [threading.Thread(target=read_stdout, daemon=True),
               threading.Thread(target=lambda: errors.extend(process.stderr), daemon=True)]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + TIMEOUT_S
    timed_out = False
    try:
        for request in requests:
            process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            process.stdin.flush()
            if "id" not in request:
                continue
            while True:
                try:
                    got = replies.get(timeout=max(deadline - time.monotonic(), 0.1))
                except queue.Empty:
                    timed_out = True
                    break
                if got is None or got == request["id"]:
                    break
            if timed_out or got is None:
                break
    except BrokenPipeError:
        pass
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            process.wait(timeout=max(deadline - time.monotonic(), 10))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            timed_out = True
        for reader in readers:
            reader.join(timeout=10)
    stderr = b"".join(errors).decode("utf-8", "replace")
    print(f"--- server stderr (exit {process.returncode})\n{stderr[-4000:]}")
    return (None if timed_out else lines), stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tools", type=int, default=None, help="the number of tools tools/list must return")
    parser.add_argument("--call", default=None, help="a tool to call")
    parser.add_argument("--arguments", default="{}", help="the JSON arguments of --call")
    parser.add_argument("--expect", action="append", default=[], help="text the instructions or the call show")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "mcp_smoke", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    if args.call:
        requests.append({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                         "params": {"name": args.call, "arguments": json.loads(args.arguments)}})
    stdout_lines, stderr = _exchange(command, requests)
    if stdout_lines is None:
        print(f"FAIL: the server did not answer within {TIMEOUT_S} s", file=sys.stderr)
        return 1
    problems: list[str] = []
    replies: dict[int, dict] = {}
    for n, raw in enumerate(stdout_lines, start=1):
        if not raw.strip():
            continue
        try:
            message = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            problems.append(f"stdout line {n} is not a protocol message: {raw[:200]!r}")
            continue
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            problems.append(f"stdout line {n} is not JSON-RPC 2.0: {raw[:200]!r}")
            continue
        if "id" in message:
            replies[message["id"]] = message
    for request in requests:
        rid = request.get("id")
        if rid is None:
            continue
        reply = replies.get(rid)
        if reply is None:
            problems.append(f"no reply to {request['method']}")
        elif "error" in reply:
            problems.append(f"{request['method']} failed: {reply['error']}")
    seen = []
    if 1 in replies and "result" in replies[1]:
        result = replies[1]["result"]
        seen.append(result.get("instructions") or "")
        print(f"server: {result.get('serverInfo')}")
    if 2 in replies and "result" in replies[2]:
        names = [t["name"] for t in replies[2]["result"]["tools"]]
        print(f"{len(names)} tools: {', '.join(names)}")
        if args.tools is not None and len(names) != args.tools:
            problems.append(f"tools/list returned {len(names)} tools, expected {args.tools}")
    if args.call and 3 in replies and "result" in replies[3]:
        result = replies[3]["result"]
        call_text = "".join(c.get("text", "") for c in result.get("content", []))
        print(f"{args.call}: {call_text[:1000]}")
        seen.append(call_text)
        if result.get("isError"):
            problems.append(f"{args.call} answered with an error: {call_text[:500]}")
    for expected in args.expect:
        if not any(expected in s for s in seen):
            problems.append(f"neither the instructions nor the call show {expected!r}")
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if not problems:
        print("OK: only protocol messages on stdout")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
