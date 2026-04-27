"""Entry point for the code search MCP server."""
from code_search_mcp.server import app


def main() -> None:
    """Start the MCP server."""
    app.run()


if __name__ == "__main__":
    main()
