"""Local natural-language SQL exploration prototype."""

from .cli import main
from .config import Settings, SettingsError

__all__ = ["Settings", "SettingsError", "main"]
