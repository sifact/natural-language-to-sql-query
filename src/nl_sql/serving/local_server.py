"""Serve the web app on this machine with the standard-library WSGI server."""

from __future__ import annotations

from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from .app import create_app

DEFAULT_PORT = 8000


class _ThreadingServer(ThreadingMixIn, WSGIServer):
    """Answer the page and its API while a slow model call is still running."""

    daemon_threads = True


class _QuietHandler(WSGIRequestHandler):
    """The app logs each request itself, so skip the default access log."""

    def log_message(self, format: str, *args: object) -> None:
        return


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
    """Serve the UI until interrupted.

    Args:
        port: Port to listen on.
        host: Interface to bind; loopback by default because the server holds the API key.

    Raises:
        SettingsError: If the model endpoint is not configured.
    """
    app = create_app()
    with make_server(host, port, app, _ThreadingServer, _QuietHandler) as server:
        shown = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        print(f"nl-sql UI running at http://{shown}:{port}  (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nnl-sql UI stopped.")
