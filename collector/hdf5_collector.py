import h5py
import numpy as np


def _memspec_to_bytes(memspec: int | str | None) -> int | None:
    if memspec is None:
        return None
    if isinstance(memspec, int):
        return memspec
    if isinstance(memspec, str):
        if memspec.endswith("KB"):
            return int(memspec[:-2]) * 1024
        if memspec.endswith("MB"):
            return int(memspec[:-2]) * 1024**2
        if memspec.endswith("GB"):
            return int(memspec[:-2]) * 1024**3
        if memspec.endswith("TB"):
            return int(memspec[:-2]) * 1024**4
        if memspec.endswith("B"):
            try:
                return int(memspec[:-1])
            except ValueError:
                pass
        raise ValueError(f"Invalid memory specification: {memspec}")


class HDF5Collector:
    """Collector interface for episodic data to HDF5 files.

    Args:
        file: The HDF5 file to write to.
        batch_size: The batch size.
        chunk: Chunk size for the data.
            if int: Chunk size in number of elements.
            if str: Chunk size in bytes.
            if None: Auto-chunking. (default: None)
        compression: Compression algorithm to use.
            if None: No compression. (default: None)
            options: gzip (best, slow), lzf (good, fast)
    """

    def __init__(
        self,
        file: h5py.File,
        batch_size: int = 1,
        chunk: int | str | None = None,
        compression: str | None = None,
    ):
        self._file = file
        self._batch_size = batch_size
        if chunk is None:
            self._chunk_info = "auto"
            self._chunk_data = True
        else:
            if isinstance(chunk, int):
                self._chunk_info = "length"
                self._chunk_size = chunk
            elif isinstance(chunk, str):
                self._chunk_info = "bytes"
                self._chunk_size = _memspec_to_bytes(chunk)
        self._compression = compression

        self._attr_cache = []
        self._max_id = max([int(key[5:]) for key in self._file.keys() if key.startswith("demo")], default=-1) + 1
        self._ids = []
        for _ in range(self._batch_size):
            self._ids.append(self._max_id)
            self._max_id += 1

    def add(self, name: str, data: np.ndarray, mask: np.ndarray | None = None) -> None:
        """Add a batch of data to the HDF5 file.

        Args:
            name: The name of the dataset.
            data: The data to add. Assumes the first axis of the data is the batch axis.
            mask: The mask on which episodes to extend. (default: None = all episodes)
        """
        if len(data) != self._batch_size:
            if mask is None:
                raise ValueError(
                    f"Data batch size ({len(data)}) does not match collector batch size ({self._batch_size})"
                )
            else:
                if not mask.sum() == len(data):
                    raise ValueError(
                        f"Data batch size ({len(data)}) is not equal to the number of True values in the mask ({mask.int().sum()})"
                    )

        data_group = self._file.require_group("data")

        data_idx = 0
        for idx, id in enumerate(self._ids):
            if mask is not None and not mask[idx]:
                continue
            key = f"demo_{id}/{name}"
            if key not in data_group.keys():
                data_group.create_dataset(
                    key,
                    data=data[data_idx : data_idx + 1],
                    maxshape=(None,) + data.shape[1:],
                    dtype=data.dtype,
                    chunks=self.get_chunking(data),
                    compression=self._compression,
                )
            else:
                data_group[key].resize(data_group[key].shape[0] + 1, axis=0)
                data_group[key].write_direct(data, np.s_[data_idx], np.s_[-1])

    def add_attribute(self, name: str | None, key: str, value: str, mask: np.ndarray | None) -> None:
        """Add an attribute to the HDF5 file.

        Args:
            name: The name of the dataset. If None, the attribute is added to the root group.
            key: The key of the attribute.
            value: The value of the attribute.
            mask: The mask on which episodes to extend. (default: None = all episodes)
        """
        self._attr_cache.append((name, key, value, mask))

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Reset the open episodes.

        Args:
            mask: mask of which episodes to reset. (default: None = all episodes)
        """
        self.flush()
        self._refresh_ids(mask)

    """ UTILS """

    def _refresh_ids(self, mask: np.ndarray | None = None) -> np.ndarray:
        for i in range(self._batch_size):
            if mask is None or mask[i]:
                self._ids[i] = self._max_id
                self._max_id += 1

    def flush(self) -> None:
        if "data" in self._file.keys():
            data_dset = self._file["data"]
        else:
            data_dset = self._file.create_group("data")

        if len(self._attr_cache) > 0:
            for name, key, value, mask in self._attr_cache:
                for idx, id in enumerate(self._ids):
                    if mask is not None and not mask[idx]:
                        continue
                    name = f"demo_{id}/{name}" if name is not None else f"demo_{id}"
                    if name not in data_dset.keys():
                        raise ValueError(f"Dataset '{name}' does not exist")
                    data_dset[name].attrs[key] = value

        self._file.flush()

        self._data_cache = []
        self._attr_cache = []
        self._ram = 0

    def get_chunking(self, batch):
        item_shape = batch.shape[1:]
        item_bytes = batch.nbytes // self._batch_size
        if self._chunk_info == "auto":
            return True
        elif self._chunk_info == "length":
            return (self._chunk_size,) + item_shape
        elif self._chunk_info == "bytes":
            return (max(self._chunk_size // item_bytes, 1),) + item_shape
        else:
            raise ValueError(f"Invalid chunk info: {self._chunk_info}")
