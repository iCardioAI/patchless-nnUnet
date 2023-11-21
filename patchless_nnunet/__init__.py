import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ENV_PNNUNET_HOME = "PATCHLESS_NNUNET_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def get_root() -> Path:
    """Resolves the root directory for the `Patchless-nnunet` package.

    Returns:
        Path to the root directory for the `Patchless-nnunet` package.
    """
    return Path(__file__).resolve().parent


def get_home() -> Path:
    """Resolves the home directory for the `Patchless-nnunet` library, used to save/cache data reusable
    across scripts/runs.

    Returns:
        Path to the home directory for the `Patchless-nnunet` library.
    """
    load_dotenv()
    home = os.getenv(ENV_PNNUNET_HOME)
    if home is None:
        user_cache_dir = os.getenv("XDG_CACHE_HOME", DEFAULT_CACHE_DIR)
        home = os.path.join(user_cache_dir, "patchless_nnunet")
    return Path(home).expanduser()


def setup_root(
    project_root_env_var: bool = True,
    dotenv: bool = True,
    pythonpath: bool = True,
    cwd: bool = False,
) -> Path:
    """Find and setup the project root.

    Args:
        project_root_env_var (bool, optional): Whether to set PROJECT_ROOT environment variable.
        dotenv (bool, optional): Whether to load `.env` file.
        pythonpath (bool, optional): Whether to add project root to pythonpath.
        cwd (bool, optional): Whether to set current working directory to project root.

    Raises:
        FileNotFoundError: If root is not found.

    Returns:
        Path: Path to project root.
    """
    # Get the project root path
    path = str(get_root())

    if not os.path.exists(path):
        raise FileNotFoundError(f"Project root path does not exist: {path}")

    # Set the `PROJECT_ROOT` that will be used in Hydra default path config
    if project_root_env_var:
        os.environ["PROJECT_ROOT"] = path

    # Load any available `.env` file
    if dotenv:
        load_dotenv()

    # Add project root to pythonpath
    if pythonpath:
        sys.path.insert(0, path)

    # Set current working directory to project root
    if cwd:
        os.chdir(path)

    return path
