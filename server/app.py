"""
server/app.py — OpenEnv entry point shim.

Loads the root-level server.py by absolute file path so that the
server/ package and the root server.py file can coexist without
Python's import system preferring the package over the module.
"""
import importlib.util
import os
import sys

# Add project root to path so root-level modules (env, models, etc.) resolve
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Load root server.py by file path, registered under a private module name
# to avoid collision with the `server` package name.
_spec = importlib.util.spec_from_file_location(
    "_dataselectenv_server",
    os.path.join(_root, "server.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_dataselectenv_server"] = _mod
_spec.loader.exec_module(_mod)

# Re-export the FastAPI app — this is what openenv and uvicorn look for.
app = _mod.app


def main() -> None:
    """Entry point required by openenv validate and [project.scripts]."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
