"""
MFAAvalonia-inspired local Web UI server and API bridge for Gakumas Demo.
"""

from demo.web.server import (
    WebSessionManager,
    create_web_server,
    run_web_server,
)

__all__ = [
    "WebSessionManager",
    "create_web_server",
    "run_web_server",
]
