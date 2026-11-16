"""不允许 pickle 的实验数组读取。"""

import numpy as np


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


