# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
import torch


def set_random_seed(seed):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
                    If not provided or set to 0, a random seed will be generated.
    """
    if not seed or seed == 0:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)  # python随机数
    np.random.seed(seed)  # np的随机数
    torch.manual_seed(seed)  # torch在cpu的随机数
    torch.cuda.manual_seed_all(seed)  # torch在gpu的随机数
    return seed
