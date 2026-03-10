"""WZRD MCP Server — MCP tool layer for the wzrd projection mapping toolkit."""

from pathlib import Path

from dotenv import load_dotenv

# Path to the secrets file — change this if you switch to a different file.
DOTENV_PATH = Path.home() / ".eve"

load_dotenv(DOTENV_PATH)  # no-op if the file doesn't exist
