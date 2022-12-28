
import torch.multiprocessing as mp
from torch.distributed import init_process_group, get_rank, get_world_size
from torch.utils.data import IterableDataset, Dataset, DataLoader, get_worker_info, DistributedSampler

NUM_SAMPLES = 16
NUM_PROCESSES = 2

def print_on_rank(*args, **kwargs):
    rank = get_rank()
    print(f"[rank={rank}]", *args, **kwargs)


class MapDataset(Dataset):

    def __len__(self):
        return NUM_SAMPLES

    def __getitem__(self, idx):
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
                    yield i
                else:
                    continue

class DataParallelIterableDataset(IterableDataset):

    def __len__(self):
        # Caveat: When using DistributedSampler, we need to know the number of samples in our dataset!
        # Hence, we need to implement `__len__`.
        return NUM_SAMPLES

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        
        world_size = get_world_size()
        process_rank = get_rank()

        sampler = DistributedSampler(self, num_replicas=(num_workers * world_size), rank=(process_rank * num_workers + worker_id), shuffle=False)

        for i in iter(sampler):
            yield i



def naive_data_parallel_with_map_dataset(rank):
    init_process_group(backend="gloo", init_method="tcp://localhost:1111", world_size=NUM_PROCESSES, rank=rank)
    
    dataloader_map_dataset = DataLoader(MapDataset(), batch_size=4, num_workers=2)
    print_on_rank(list(dataloader_map_dataset))
    

def fixed_data_parallel_with_map_dataset(rank):
    init_process_group(backend="gloo", init_method="tcp://localhost:1111", world_size=NUM_PROCESSES, rank=rank)

    dataset = MapDataset()
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader_map_dataset = DataLoader(dataset, batch_size=4, num_workers=2, sampler=sampler)
    print_on_rank(list(dataloader_map_dataset))


def naive_data_parallel_with_iterable_dataset(rank):
    init_process_group(backend="gloo", init_method="tcp://localhost:1111", world_size=NUM_PROCESSES, rank=rank)
    
    dataloader_iterable_dataset = DataLoader(MultiprocessingIterableDataset(), batch_size=4, num_workers=2)
    print_on_rank(list(dataloader_iterable_dataset))


def fixed_data_parallel_with_iterable_dataset(rank):
    init_process_group(backend="gloo", init_method="tcp://localhost:1111", world_size=NUM_PROCESSES, rank=rank)
    
    # Note, we can't just fix the issue by adding a DistributedSampler to the DataLoader like we did for the MapDataset.
    # We would get an error: "ValueError: DataLoader with IterableDataset: expected unspecified sampler option"
    # We need to integrate the DistributedSampler directly into the IterableDataset!
    dataloader_iterable_dataset = DataLoader(DataParallelIterableDataset(), batch_size=4, num_workers=2)
    print_on_rank(list(dataloader_iterable_dataset))



def main():
    
    print("Map Dataset and Distributed Sampler; num_workers: 2, batch_size: 4")
    # Observe duplicated data across the different processes
    mp.start_processes(naive_data_parallel_with_map_dataset, nprocs=NUM_PROCESSES, start_method="fork")

    print("\n")
    
    print("Map Dataset and Distributed Sampler; num_workers: 2, batch_size: 4")
    mp.start_processes(fixed_data_parallel_with_map_dataset, nprocs=NUM_PROCESSES, start_method="fork")

    print("\n")

    print("Maive Multiprocessing-friendly Iterable Dataset; num_workers: 2, batch_size: 4")
    # Observe duplicated data across the different processes
    mp.start_processes(naive_data_parallel_with_iterable_dataset, nprocs=NUM_PROCESSES, start_method="fork")

    print("\n")

    print("DDP-friendly + Multiprocessing-friendly Iterable Dataset; num_workers: 2, batch_size: 4")
    # No duplicated data
    mp.start_processes(fixed_data_parallel_with_iterable_dataset, nprocs=NUM_PROCESSES, start_method="fork")


if __name__ == "__main__":
    main()
