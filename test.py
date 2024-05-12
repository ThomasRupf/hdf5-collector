from collector.hdf5_collector import HDF5Collector


def test_hdf5_collector():
    import h5py
    import numpy as np

    data = {
        "demo_0": {
            "a": np.random.rand(100),
            "b": np.random.rand(100, 100),
        },
        "demo_1": {
            "a": np.random.rand(99),
            "b": np.random.rand(99, 100),
        },
        "demo_2": {
            "a": np.random.rand(101),
            "b": np.random.rand(101, 100),
        },
        "demo_3": {
            "a": np.random.rand(102),
            "b": np.random.rand(102, 100),
        },
    }

    with h5py.File("test.h5", "w") as file:
        collector = HDF5Collector(file, chunk="4KB", max_ram="128KB")

        for i in range(99):
            a_batch = np.stack((data["demo_0"]["a"][i], data["demo_1"]["a"][i]))
            b_batch = np.stack((data["demo_0"]["b"][i], data["demo_1"]["b"][i]))
            collector.add_batched("a", a_batch)
            collector.add_batched("b", b_batch)

        collector.reset(np.array([False, True]))
        print(collector._ids)

        a_batch = np.stack((data["demo_0"]["a"][99], data["demo_2"]["a"][0]))
        b_batch = np.stack((data["demo_0"]["b"][99], data["demo_2"]["b"][0]))
        collector.add_batched("b", b_batch)
        collector.add_batched("a", a_batch)

        collector.reset(np.array([True, False]))
        print(collector._ids)

        a_batch = np.stack((data["demo_3"]["a"][0],))
        b_batch = np.stack((data["demo_3"]["b"][0],))

        collector.add_batched("a", a_batch, mask=np.array([False, True]))
        collector.add_batched("b", b_batch, mask=np.array([False, True]))

        for i in range(1, 101):
            a_batch = np.stack((data["demo_2"]["a"][i], data["demo_3"]["a"][i]))
            b_batch = np.stack((data["demo_2"]["b"][i], data["demo_3"]["b"][i]))
            collector.add_batched("a", a_batch)
            collector.add_batched("b", b_batch)

        collector.reset(np.array([True, False]))
        print(collector._ids)

        a_batch = np.stack((data["demo_3"]["a"][101],))
        b_batch = np.stack((data["demo_3"]["b"][101],))
        collector.add_batched("a", a_batch, mask=np.array([False, True]))
        collector.add_batched("b", b_batch, mask=np.array([False, True]))

        collector.reset()

    with h5py.File("test.h5", "r") as file:

        def print_rec(x):
            if isinstance(x, h5py.Group):
                print(f"Group: {x.name}")
                for key in x.keys():
                    print_rec(x[key])
            elif isinstance(x, h5py.Dataset):
                print(f"Dataset: {x.name}", x.shape, x.dtype)

        print_rec(file)

        for demo in data.keys():
            for key in data[demo].keys():
                gt = data[demo][key]
                file_data = file[f"data/{demo}/{key}"]
                print(f"{demo}/{key}", gt.shape, gt.dtype)
                print(file_data.name, file_data.shape, file_data.dtype)
                assert gt.shape == file_data.shape, f"Shape mismatch for {demo}/{key}"
                assert gt.dtype == file_data.dtype, f"Dtype mismatch for {demo}/{key}"

    print("All tests passed!")


if __name__ == "__main__":
    test_hdf5_collector()
