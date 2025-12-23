#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BrainStorm MCP Servers Startup Script.

Запускает все MCP серверы:
- BraineMemory на порту 8001 (SSE)
- LegalOps на порту 8002 (SSE)

Agent подключается к ним через HTTP/SSE.

Использование:
    python -m integration.start_all

    # Только BraineMemory
    python -m integration.start_all --braine-only

    # Только LegalOps
    python -m integration.start_all --legal-only

    # С кастомными портами
    python -m integration.start_all --braine-port 9001 --legal-port 9002
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base path
BASE_PATH = Path(__file__).parent.parent


def start_braine_server(port: int = 8001, transport: str = "sse") -> subprocess.Popen:
    """Start BraineMemory MCP server."""
    cwd = BASE_PATH / "BraineMemory"
    cmd = [
        sys.executable, "-m", "src.mcp.server",
        "--transport", transport,
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    logger.info(f"Starting BraineMemory on port {port}...")
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def start_legal_server(port: int = 8002, transport: str = "sse") -> subprocess.Popen:
    """Start LegalOps MCP server."""
    cwd = BASE_PATH / "LegalOps"
    cmd = [
        sys.executable, "-m", "legalops.mcp.server",
        "--transport", transport,
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    logger.info(f"Starting LegalOps on port {port}...")
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


async def log_output(name: str, proc: subprocess.Popen):
    """Stream process output to logger."""
    while proc.poll() is None:
        if proc.stdout:
            line = proc.stdout.readline()
            if line:
                logger.info(f"[{name}] {line.decode().strip()}")
        await asyncio.sleep(0.01)


async def main(
    braine_port: int = 8001,
    legal_port: int = 8002,
    braine_only: bool = False,
    legal_only: bool = False,
    transport: str = "sse",
):
    """Main entry point."""
    processes = []
    tasks = []

    try:
        # Start servers
        if not legal_only:
            braine_proc = start_braine_server(braine_port, transport)
            processes.append(("BraineMemory", braine_proc))
            tasks.append(asyncio.create_task(log_output("braine", braine_proc)))

        if not braine_only:
            legal_proc = start_legal_server(legal_port, transport)
            processes.append(("LegalOps", legal_proc))
            tasks.append(asyncio.create_task(log_output("legal", legal_proc)))

        # Wait a bit for servers to start
        await asyncio.sleep(2)

        # Print status
        print("\n" + "=" * 60)
        print("BrainStorm MCP Servers")
        print("=" * 60)

        if not legal_only:
            print(f"\n  BraineMemory: http://localhost:{braine_port}")
            print(f"    - Health: http://localhost:{braine_port}/health")
            print(f"    - SSE: http://localhost:{braine_port}/sse")

        if not braine_only:
            print(f"\n  LegalOps: http://localhost:{legal_port}")
            print(f"    - Health: http://localhost:{legal_port}/health")
            print(f"    - SSE: http://localhost:{legal_port}/sse")

        print("\n" + "-" * 60)
        print("Agent connection URLs:")
        if not legal_only:
            print(f"  BRAINE_MCP_URL=http://localhost:{braine_port}/sse")
        if not braine_only:
            print(f"  LEGAL_MCP_URL=http://localhost:{legal_port}/sse")
        print("=" * 60)
        print("\nPress Ctrl+C to stop all servers\n")

        # Wait for interrupt
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Cleanup
        for name, proc in processes:
            if proc.poll() is None:
                logger.info(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Start BrainStorm MCP Servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--braine-port",
        type=int,
        default=8001,
        help="BraineMemory server port (default: 8001)",
    )
    parser.add_argument(
        "--legal-port",
        type=int,
        default=8002,
        help="LegalOps server port (default: 8002)",
    )
    parser.add_argument(
        "--braine-only",
        action="store_true",
        help="Start only BraineMemory server",
    )
    parser.add_argument(
        "--legal-only",
        action="store_true",
        help="Start only LegalOps server",
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "http"],
        default="sse",
        help="Transport type (default: sse)",
    )

    args = parser.parse_args()

    asyncio.run(main(
        braine_port=args.braine_port,
        legal_port=args.legal_port,
        braine_only=args.braine_only,
        legal_only=args.legal_only,
        transport=args.transport,
    ))


if __name__ == "__main__":
    cli()
