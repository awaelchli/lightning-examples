from torch.utils.data import IterableDataset, Dataset, DataLoader, get_worker_info

NUM_SAMPLES = 16


class SimpleMapDataset(Dataset):

    def __len__(self):
        return NUM_SAMPLES

    def __getitem__(self, idx):
        return idx



class SimpleIterableDataset(IterableDataset):

    def __iter__(self):
        for i in range(NUM_SAMPLES):
            yield i


class MultiprocessingMapDataset(Dataset):

    def __len__(self):
        return NUM_SAMPLES

    def __getitem__(self, idx):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            print(f"Worker {worker_id} fetches sample {idx}")
        return idx


class MultiprocessingIterableDataset(IterableDataset):

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Case 1: Single process
            for i in range(NUM_SAMPLES):
                yield i
        else:
            # Case 2: Multiprocessing (num_workers > 0)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            for i in range(NUM_SAMPLES):
                if i % num_workers == worker_id:
                    print(f"Worker {worker_id} fetches sample {i}")
                    yield i
                else:
                    continue


def main():

    print("Map Dataset; num_workers: 0, batch_size: 1")
    dataloader_map_dataset = DataLoader(SimpleMapDataset())
    print(list(dataloader_map_dataset))
    
    print("\n")

    print("Iterable Dataset; num_workers: 0, batch_size: 1")
    dataloader_iterable_dataset = DataLoader(SimpleIterableDataset())
    print(list(dataloader_iterable_dataset))

    print("\n")

    print("Map Dataset; num_workers: 0, batch_size: 4")
    dataloader_map_dataset = DataLoader(SimpleMapDataset(), batch_size=4)
    print(list(dataloader_map_dataset))
    
    print("\n")

    print("Iterable Dataset; num_workers: 0, batch_size: 4")
    dataloader_iterable_dataset = DataLoader(SimpleIterableDataset(), batch_size=4)
    print(list(dataloader_iterable_dataset))

    print("\n")

    print("Map Dataset; num_workers: 2, batch_size: 4")
    dataloader_map_dataset = DataLoader(MultiprocessingMapDataset(), batch_size=4, num_workers=2)
    print(list(dataloader_map_dataset))
    
    print("\n")

    print("Naive Iterable Dataset; num_workers: 2, batch_size: 4")
    # Observe the duplicated data!
    dataloader_iterable_dataset = DataLoader(SimpleIterableDataset(), batch_size=4, num_workers=2)
    print(list(dataloader_iterable_dataset))

    print("\n")
    
    print("Multiprocessing-friendly Iterable Dataset; num_workers: 2, batch_size: 4")
    # No duplicated data
    dataloader_iterable_dataset = DataLoader(MultiprocessingIterableDataset(), batch_size=4, num_workers=2)
    print(list(dataloader_iterable_dataset))
        

if __name__ == "__main__":
    main()
