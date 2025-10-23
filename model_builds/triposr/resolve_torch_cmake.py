"""Helper script to expose Torch CMake locations for Docker builds."""

from __future__ import annotations

import pathlib

import torch


def main() -> None:
    prefix = torch.utils.cmake_prefix_path
    torch_dir = pathlib.Path(prefix) / "Torch"
    # Emit in shell-friendly `KEY=value` lines so RUN can `eval` them.
    # Using print without extra spaces keeps parsing simple.
    print(f"CMAKE_PREFIX_PATH={prefix}")
    print(f"Torch_DIR={torch_dir}")


if __name__ == "__main__":
    main()
