"""Module-level WSGI ``app`` for hosts that import one, such as Vercel."""

from .app import create_app

app = create_app()
