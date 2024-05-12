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
        chunk: Chunk size for the data.
            if int: Chunk size in number of elements.
            if str: Chunk size in bytes.
            if None: Auto-chunking. (default: None)
        compression: Compression algorithm to use.
            if None: No compression. (default: None)
            options: gzip (best, slow), lzf (good, fast)
        max_ram: Maximum amount of RAM to use.
            if int: Maximum amount of RAM in bytes.
            if str: Maximum amount of RAM in bytes.
            if None: (default: 128KB)
    """

    def __init__(
        self,
        file: h5py.File,
        chunk: int | str | None = None,
        compression: str | None = None,
        max_ram: int | str | None = "128KB",
    ):
        self._file = file
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
        self._max_ram = _memspec_to_bytes(max_ram)
        self._ram = 0

        self._data_cache = []
        self._attr_cache = []
        self._ids = []
        self._max_id = 0  # TODO: this assumes empty file!

    def add_batched(self, name: str, data: np.ndarray, mask: np.ndarray | None = None) -> None:
        """Add a batch of data to the HDF5 file.

        Args:
            name: The name of the dataset.
            data: The data to add. Assumes the first axis of the data is the batch axis.
            mask: The mask on which episodes to extend. (default: None = all episodes)
        """

        self._ram += data.nbytes
        self._ram += mask.nbytes if mask is not None else 0

        self._data_cache.append((name, data, mask))

        if self._ram > self._max_ram:
            self.flush()

    def add_single(self, name: str, data: np.ndarray) -> None:
        """Add a single data point to the HDF5 file.

        Args:
            name: The name of the dataset.
            data: The data to add.
        """
        self.add_batched(name, data[np.newaxis], None)

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

        self._close_ids(mask)

    """ UTILS """

    def _close_ids(self, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones(len(self._ids), dtype=bool)
        self._ids = [None if mask[idx] else i for idx, i in enumerate(self._ids)]

    def flush(self) -> None:
        if "data" in self._file.keys():
            data_dset = self._file["data"]
        else:
            data_dset = self._file.create_group("data")

        if len(self._data_cache) > 0:
            # since we flush on resets, we can assume all data has the same batch size
            B = self._data_cache[0][1].shape[0]
            B_mask = self._data_cache[0][2].shape[0] if self._data_cache[0][2] is not None else B
            B = max(B, B_mask)  # if mask exists, it contains the true batch_size

            # add new ids if necessary
            for i in range(B):
                if len(self._ids) <= i:
                    self._ids.append(None)
                if self._ids[i] is None:
                    self._ids[i] = self._max_id
                    self._max_id += 1

            def get_chunking(bytes, shape):
                if self._chunk_info == "auto":
                    return True
                if self._chunk_info == "length":
                    return (self._chunk_size, *shape)
                if self._chunk_info == "bytes":
                    return (max(self._chunk_size // bytes, 1), *shape)
                raise ValueError(f"Invalid chunking info: {self._chunk_info}")

            # collect metadata for new data
            new_data_meta = dict()
            for name, data, mask in self._data_cache:
                item_shape = data.shape[1:]
                item_dtype = data.dtype
                item_bytes = int(data.nbytes / B)

                if name not in new_data_meta.keys():
                    new_data_meta[name] = {
                        "meta": {
                            "shape": item_shape,
                            "dtype": item_dtype,
                            "chunking": get_chunking(item_bytes, item_shape),
                        },
                        "count": np.where(mask, 1, 0) if mask is not None else np.ones(B, dtype=int),
                    }
                else:
                    new_data_meta[name]["count"] += np.where(mask, 1, 0) if mask is not None else 1
                    if new_data_meta[name]["meta"]["shape"] != item_shape:
                        raise ValueError(f"Shape mismatch for dataset '{name}'")
                    if new_data_meta[name]["meta"]["dtype"] != item_dtype:
                        raise ValueError(f"Dtype mismatch for dataset '{name}'")

            # create missing datasets and allocate space
            data_time_idx = dict()
            for idx, id in enumerate(self._ids):
                for name, meta in new_data_meta.items():
                    name = f"demo_{id}/{name}"
                    if meta["count"][idx] > 0:
                        if name not in data_dset.keys():
                            data_dset.create_dataset(
                                name,
                                shape=(0, *meta["meta"]["shape"]),
                                maxshape=(None, *meta["meta"]["shape"]),
                                dtype=meta["meta"]["dtype"],
                                chunks=meta["meta"]["chunking"],
                                compression=self._compression,
                            )
                        data_time_idx[name] = data_dset[name].shape[0]
                        data_dset[name].resize(data_dset[name].shape[0] + meta["count"][idx], axis=0)

            # write data
            for name, data, mask in self._data_cache:
                data_idx = 0
                for idx, id in enumerate(self._ids):
                    if mask is not None and not mask[idx]:
                        continue
                    data_name = f"demo_{id}/{name}"
                    h5py_dset: h5py.Dataset = data_dset[data_name]
                    h5py_dset.write_direct(data, np.s_[data_idx], np.s_[data_time_idx[data_name]])
                    data_time_idx[data_name] += 1
                    data_idx += 1

        if len(self._attr_cache) > 0:
            # add new ids if necessary
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
