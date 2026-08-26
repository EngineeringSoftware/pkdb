from typing import List, Optional, Union

import numpy as np

from .views import View, ViewType, array


class BinOp:
    def __init__(
        self,
        keys: View,
        max_bins: Union[int, List[int]],
        min_value: Union[float, List[float]],
        max_value: Union[float, List[float]],
    ):
        self.keys = keys
        self.max_bins = max_bins
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def get_type(dim: int, key_view_type: str) -> str:
        return f"Kokkos::BinOp{dim}D<{key_view_type}>"

    def _keys_array(self) -> np.ndarray:
        return self.keys.data if isinstance(self.keys, ViewType) else np.asarray(self.keys)

    def num_bins(self) -> int:
        raise NotImplementedError

    def bin_indices(self) -> np.ndarray:
        """Flat bin index per key, used by the interpreted (Debug) execution path."""
        raise NotImplementedError


class BinOp1D(BinOp):
    def __init__(self, keys: View, max_bins: int, min_value: float, max_value: float):
        super().__init__(keys, max_bins, min_value, max_value)

    def num_bins(self) -> int:
        return self.max_bins

    def bin_indices(self) -> np.ndarray:
        keys_arr = self._keys_array()
        span = self.max_value - self.min_value
        scale = self.max_bins / span if span > 0 else 0.0
        idx = np.floor((keys_arr - self.min_value) * scale).astype(np.int64)
        return np.clip(idx, 0, self.max_bins - 1)


class BinOp3D(BinOp):
    def __init__(
        self,
        keys: View,
        max_bins: List[int],
        min_value: List[float],
        max_value: List[float],
    ):
        super().__init__(keys, max_bins, min_value, max_value)

    def num_bins(self) -> int:
        nbx, nby, nbz = self.max_bins
        return nbx * nby * nbz

    def bin_indices(self) -> np.ndarray:
        """Row-major flat index: ix * nby * nbz + iy * nbz + iz."""
        keys_arr = self._keys_array()
        nbx, nby, nbz = self.max_bins
        idx3 = np.empty((keys_arr.shape[0], 3), dtype=np.int64)
        for d, n in enumerate((nbx, nby, nbz)):
            span = self.max_value[d] - self.min_value[d]
            scale = n / span if span > 0 else 0.0
            col = np.floor((keys_arr[:, d] - self.min_value[d]) * scale).astype(np.int64)
            idx3[:, d] = np.clip(col, 0, n - 1)
        return idx3[:, 0] * (nby * nbz) + idx3[:, 1] * nbz + idx3[:, 2]


class BinSort:
    def __init__(self, keys: View, bin_op: BinOp, sort_within_bins: bool = False):
        self.keys = keys
        self.bin_op = bin_op
        self.sort_within_bins = sort_within_bins
        self._permute_vector: Optional[np.ndarray] = None
        self._bin_count: Optional[np.ndarray] = None
        self._bin_offsets: Optional[np.ndarray] = None

    @staticmethod
    def get_type(key_view_type: str, bin_op_type: str, space: str) -> str:
        return f"Kokkos::BinSort<{key_view_type},{bin_op_type},{space},int>"

    def sort(self, values: View) -> None:
        if self._permute_vector is None:
            raise RuntimeError("create_permute_vector() must be called before sort()")
        data = values.data if isinstance(values, ViewType) else values
        n = len(self._permute_vector)
        data[:n] = data[:n][self._permute_vector]

    def get_bin_count(self) -> View:
        return array(self._bin_count)

    def get_bin_offsets(self) -> View:
        return array(self._bin_offsets)

    def get_permute_vector(self) -> View:
        return array(self._permute_vector)

    def create_permute_vector(self) -> None:
        """Counting sort by bin index (interpreted-execution fallback for Kokkos::BinSort)."""
        bin_ids = self.bin_op.bin_indices()
        n_bins = self.bin_op.num_bins()

        counts = np.bincount(bin_ids, minlength=n_bins).astype(np.int32)
        offsets = np.zeros(n_bins, dtype=np.int32)
        if n_bins > 1:
            offsets[1:] = np.cumsum(counts)[:-1]

        # Stable sort so ties (same bin) keep their original relative order,
        # matching sort_within_bins=False semantics.
        order = np.argsort(bin_ids, kind="stable").astype(np.int32)

        self._bin_count = counts
        self._bin_offsets = offsets
        self._permute_vector = order
