#!/usr/bin/env python3
"""Run the official LeRobot trainer with an optional immutable training index view."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from runtime.training.canonical_index_view import apply_training_index_manifest


def _extract_manifest(argv: list[str]) -> tuple[Path | None, list[str]]:
    result: list[str] = [argv[0]]
    manifest: Path | None = None
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--training-index-manifest":
            if index + 1 >= len(argv):
                raise SystemExit("--training-index-manifest requires a path")
            value = argv[index + 1]
            index += 2
        elif token.startswith("--training-index-manifest="):
            value = token.split("=", 1)[1]
            index += 1
        else:
            result.append(token)
            index += 1
            continue
        if manifest is not None:
            raise SystemExit("--training-index-manifest may only be supplied once")
        manifest = Path(value).expanduser().resolve()
    return manifest, result


def main() -> None:
    manifest, passthrough_argv = _extract_manifest(sys.argv)
    from lerobot.scripts import lerobot_train

    if manifest is not None:
        official_factory = lerobot_train.make_train_eval_datasets

        def indexed_factory(cfg):
            train_dataset, eval_dataset = official_factory(cfg)
            view = apply_training_index_manifest(train_dataset, manifest)
            logging.info(
                "Validated training index manifest %s: %d samples, %d episodes, %d excluded",
                manifest,
                view.validation.sample_count,
                view.validation.episode_count,
                view.validation.excluded_count,
            )
            return view, eval_dataset

        lerobot_train.make_train_eval_datasets = indexed_factory

    sys.argv = passthrough_argv
    lerobot_train.main()


if __name__ == "__main__":
    main()
