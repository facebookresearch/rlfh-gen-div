"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="rlvsil",
    packages=find_packages(),
    version="0.0.1",
    requires=["pytorch", "transformers", "numpy", "tqdm", "datasets", "rouge", "wandb", "sacrebleu", "coolname"],
)
