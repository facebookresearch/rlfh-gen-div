"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import torch


def get_device(cuda_id: Optional[int] = None):
    cuda_dev = f"cuda:{cuda_id}" if cuda_id is not None else "cuda"
    return torch.device(cuda_dev if torch.cuda.is_available() else "cpu")
