"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""'Logging' utilities."""

import csv
import json
import logging
import os
import time


def log_to_file(localdir, _state={}, **fields):  # noqa: B006
    """Incrementally write logs.tsv into pwd."""
    if "writer" not in _state:
        path = os.path.join(localdir, "logs.jsonl")  # Could infer FLAGS if we had them.

        _state["file"] = open(path, "a", buffering=1)  # Line buffering.
        logging.info("Appending logs to %s", path)

    file = _state["file"]
    if file is not None:
        file.write(json.dumps(fields, sort_keys=True) + "\n")


def log_to_file_global(localdir, _state={}, **fields):  # noqa: B006
    """Incrementally write logs_global.tsv into pwd."""
    if "writer" not in _state:
        path = os.path.join(localdir, "logs_global.tsv")  # Could infer FLAGS if we had them.

        writeheader = not os.path.exists(path)
        fieldnames = list(fields.keys())

        _state["file"] = open(path, "a", buffering=1)  # Line buffering.
        _state["writer"] = csv.DictWriter(_state["file"], fieldnames, delimiter="\t")
        if writeheader:
            _state["writer"].writeheader()
            logging.info("Writing logs to %s", path)
        else:
            logging.info("Appending logs to %s", path)

    writer = _state["writer"]
    if writer is not None:
        writer.writerow(fields)


def symlink_path(target, symlink, force=False):
    try:
        if os.path.islink(symlink) or (force and os.path.exists(symlink)):
            os.remove(symlink)
        if not os.path.exists(symlink):
            os.symlink(target, symlink)
            return True
    except OSError:
        # os.remove() or os.symlink() raced. Don't do anything.
        pass
    return False


def write_metadata(localdir, srcdir, **kwargs):
    """Write meta.json file with some information on our setup."""
    if not localdir:
        return

    metadata = {
        "env": os.environ.copy(),
        "date_start": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata.update(kwargs)

    if False:
        try:
            import git
        except ImportError:
            logging.warning("Couldn't import gitpython module; install it with `pip install gitpython`.")
        else:
            try:
                repo = git.Repo(path=srcdir, search_parent_directories=True)
                metadata["git"] = {
                    "commit": repo.commit().hexsha,
                    "is_dirty": repo.is_dirty(),
                    "path": repo.git_dir,
                }
                if not repo.head.is_detached:
                    metadata["git"]["branch"] = repo.active_branch.name
            except git.InvalidGitRepositoryError:
                pass

        if "git" not in metadata:
            logging.warning("Couldn't determine git data.")

    symlink = os.path.join(localdir, "meta.json")
    filename = "%s.%s.%d" % (symlink, time.strftime("%Y%m%d-%H%M%S"), os.getpid())

    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    symlink_path(os.path.relpath(filename, start=localdir), symlink)
