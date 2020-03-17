### Get the online reading of files to work.

import re, gzip, pickle, time
from multiprocessing import Queue, Lock
import threading
import os
import numpy as np
from glove_inner import train_glove

class Glove(object):
    def __init__(self, cooccurence, vocab_size, alpha=0.75, x_max=100.0, d=50, seed=1234, dtype=np.float64):
        """
        Glove model for obtaining dense embeddings from a
        co-occurence (sparse) matrix.
        """
        assert(dtype == np.float64 or dtype == np.float32), "Dtype must be float32 or float64"
        self.alpha           = alpha
        self.x_max           = x_max
        self.d               = d

        self.dtype = dtype

        self.block_size = 16 if self.dtype == np.float64 else 12

        if cooccurence is dict:
            self.cooccurence     = cooccurence
            self.cooccurence_path = None
        else:
            self.cooccurence = None
            self.cooccurence_path = cooccurence

        self.seed            = seed
        np.random.seed(seed)
        self.W               = np.random.uniform(-0.5/d, 0.5/d, (vocab_size, d)).astype(dtype)
        self.ContextW        = np.random.uniform(-0.5/d, 0.5/d, (vocab_size, d)).astype(dtype)
        self.b               = np.random.uniform(-0.5/d, 0.5/d, (vocab_size, 1)).astype(dtype)
        self.ContextB        = np.random.uniform(-0.5/d, 0.5/d, (vocab_size, 1)).astype(dtype)
        self.gradsqW         = np.ones_like(self.W, dtype=dtype)
        self.gradsqContextW  = np.ones_like(self.ContextW, dtype=dtype)
        self.gradsqb         = np.ones_like(self.b, dtype=dtype)
        self.gradsqContextB  = np.ones_like(self.ContextB, dtype=dtype)

    def train_from_file(self, step_size=0.05, workers = 9, batch_size=50, verbose=False):
        # Worker function:
        total_error = [0.0]
        total_done  = [0]
        def worker_train():
            error = np.zeros(1, dtype = np.float64)
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                train_glove(self, job, step_size, error)
                with lock:
                    total_error[0] += error[0]
                    total_done[0] += len(job[0])
                    if verbose:
                        if total_done[0] % 1000 == 0:
                            print("Completed %.3f%%\r" % (100.0 * total_done[0] / total_els))
                error[0] = 0.0

        # Create workers
        workers_threads = [threading.Thread(target=worker_train) for _ in range(workers)]
        for thread in workers_threads:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # Batch co-occurence pieces
        batch_length = 0
        batch = []
        num_examples = 0
        for key in self.cooccurence:
            for subkey in self.cooccurence[key]:
                batch.append((key, subkey, self.cooccurence[key][subkey]))
                batch_length += 1
                if batch_length >= batch_size:
                    jobs.put(
                        (
                            np.array([k for k,s,c in batch], dtype=np.int32),
                            np.array([s for k,s,c in batch], dtype=np.int32),
                            np.array([c for k,s,c in batch], dtype=np.float64)
                        )
                    )
                    num_examples += len(batch)
                    batch = []
                    batch_length = 0
        if len(batch) > 0:
            jobs.put(
                (
                    np.array([k for k,s,c in batch], dtype=np.int32),
                    np.array([s for k,s,c in batch], dtype=np.int32),
                    np.array([c for k,s,c in batch], dtype=np.float64)
                )
            )
            num_examples += len(batch)
            batch = []
            batch_length = 0

        for _ in range(workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers_threads:
            thread.join()

        return total_error[0] / num_examples


    def train_from_dict(self, step_size=0.05, workers = 9, batch_size=50, verbose=False):
        # Worker function:
        total_error = [0.0]
        total_done  = [0]
        def worker_train():
            error = np.zeros(1, dtype = np.float64)
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                train_glove(self, job, step_size, error)
                with lock:
                    total_error[0] += error[0]
                    total_done[0] += len(job[0])
                    if verbose:
                        if total_done[0] % 1000 == 0:
                            print("Completed %.3f%%\r" % (100.0 * total_done[0] / total_els))
                error[0] = 0.0

        # Create workers
        workers_threads = [threading.Thread(target=worker_train) for _ in range(workers)]
        for thread in workers_threads:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # Batch co-occurence pieces
        batch_length = 0
        batch = []
        num_examples = 0
        for key in self.cooccurence:
            for subkey in self.cooccurence[key]:
                batch.append((key, subkey, self.cooccurence[key][subkey]))
                batch_length += 1
                if batch_length >= batch_size:
                    jobs.put(
                        (
                            np.array([k for k,s,c in batch], dtype=np.int32),
                            np.array([s for k,s,c in batch], dtype=np.int32),
                            np.array([c for k,s,c in batch], dtype=np.float64)
                        )
                    )
                    num_examples += len(batch)
                    batch = []
                    batch_length = 0
        if len(batch) > 0:
            jobs.put(
                (
                    np.array([k for k,s,c in batch], dtype=np.int32),
                    np.array([s for k,s,c in batch], dtype=np.int32),
                    np.array([c for k,s,c in batch], dtype=np.float64)
                )
            )
            num_examples += len(batch)
            batch = []
            batch_length = 0

        for _ in range(workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers_threads:
            thread.join()

        return total_error[0] / num_examples

    def train(self, step_size=0.05, workers = 9, batch_size=50, verbose=False):
        jobs = Queue(maxsize=2 * workers)
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        if self.cooccurence is dict:
            total_els = 0
            for key in self.cooccurence:
                for subkey in self.cooccurence[key]:
                    total_els += 1
        else:
            statinfo = os.stat(self.cooccurence_path)
            total_els = int((statinfo.st_size / self.block_size))

        print("%d total examples" % (total_els,))

        # Worker function:
        total_error = [0.0]
        total_done  = [0]
        def worker_train():
            error = np.zeros(1, dtype = np.float64)
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                train_glove(self, job, step_size, error)
                with lock:
                    total_error[0] += error[0]
                    total_done[0] += len(job[0])
                    if verbose:
                        if total_done[0] % 1000 == 0:
                            print("Completed %.3f%%\r" % (100.0 * total_done[0] / total_els))
                error[0] = 0.0

        # Create workers
        workers_threads = [threading.Thread(target=worker_train) for _ in range(workers)]
        for thread in workers_threads:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # Batch co-occurence pieces
        batch_length = 0
        batch = []
        num_examples = 0
        for key in self.cooccurence:
            for subkey in self.cooccurence[key]:
                batch.append((key, subkey, self.cooccurence[key][subkey]))
                batch_length += 1
                if batch_length >= batch_size:
                    jobs.put(
                        (
                            np.array([k for k,s,c in batch], dtype=np.int32),
                            np.array([s for k,s,c in batch], dtype=np.int32),
                            np.array([c for k,s,c in batch], dtype=np.float64)
                        )
                    )
                    num_examples += len(batch)
                    batch = []
                    batch_length = 0
        if len(batch) > 0:
            jobs.put(
                (
                    np.array([k for k,s,c in batch], dtype=np.int32),
                    np.array([s for k,s,c in batch], dtype=np.int32),
                    np.array([c for k,s,c in batch], dtype=np.float64)
                )
            )
            num_examples += len(batch)
            batch = []
            batch_length = 0

        for _ in range(workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers_threads:
            thread.join()

        return total_error[0] / num_examples
