import sys
import builtins

# Redirect all print() calls to stderr to prevent corrupting MCP JSONRPC on stdout
original_print = builtins.print
def safe_print(*args, **kwargs):
    if kwargs.get('file') in (None, sys.stdout):
        kwargs['file'] = sys.stderr
    original_print(*args, **kwargs)

builtins.print = safe_print

from amap_mcp_server import main
if __name__ == '__main__':
    sys.exit(main())
