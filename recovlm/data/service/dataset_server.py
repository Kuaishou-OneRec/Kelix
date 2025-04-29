from __future__ import annotations
import asyncio
import threading
import numpy as np
import random
import time
import heapq
from typing import Any, Dict, List, Optional

from grpc import aio
from dataset_pb2 import BatchSequenceInfo, SelectedValue
from dataset_pb2_grpc import DatasetServiceServicer, add_DatasetServiceServicer_to_server


def find_closest_numbers(lists: List[List[int]]) -> List[int]:
    K = len(lists)
    pointers = [0] * K  # Track current position in each list
    heap = []
    current_max = -float('inf')

    # Initialize heap with the first element of each list
    for i in range(K):
        val = lists[i][0]
        heapq.heappush(heap, (val, i))
        current_max = max(current_max, val)

    min_range = float('inf')
    result = [lists[i][0] for i in range(K)]  # Initialize with first elements

    while True:
        current_min, list_idx = heapq.heappop(heap)
        current_range = current_max - current_min

        # Update result if a smaller range is found
        if current_range < min_range:
            min_range = current_range
            result = [lists[i][pointers[i]] for i in range(K)]

        # Move the pointer in the list that had the current_min
        pointers[list_idx] += 1
        if pointers[list_idx] >= len(lists[list_idx]):
            break  # This list is exhausted

        # Insert next element from the same list
        next_val = lists[list_idx][pointers[list_idx]]
        heapq.heappush(heap, (next_val, list_idx))
        current_max = max(current_max, next_val)
    return result


class DataCollector:
    def __init__(self, world_size: int):
        self.world_size: int = world_size
        self.data: Dict[int, List[int]] = {}
        self.event: asyncio.Event = asyncio.Event()
        self.result: Optional[Dict[int, Any]] = None

    async def add_data(self, client_id: int, value: List[int]) -> bool:
        assert client_id not in self.data
        self.data[client_id] = value
        if len(self.data) == self.world_size:
            values = [self.data[i] for i in sorted(self.data)]
            result_index = find_closest_numbers(values)
            self.result = {cid: index for cid, index in zip(
                sorted(self.data), result_index)}
            self.event.set()
            return True
        return False


class BatchManager:
    def __init__(self, world_size: int):
        self.world_size: int = world_size
        self.collector: Optional[DataCollector] = None
        self.lock: asyncio.Lock = asyncio.Lock()
        
    async def get_collector(self) -> DataCollector:
        async with self.lock:
            if self.collector is None or self.collector.result is not None:
                self.collector = DataCollector(self.world_size)
            return self.collector    


class DatasetServiceImpl(DatasetServiceServicer):
    def __init__(self, world_size: int):
        self.manager: BatchManager = BatchManager(world_size)

    async def BalanceSequence(self, request: BatchSequenceInfo, context: aio.ServiceContext):
        print(f'receive: client_id={request.client_id}')
        collector = await self.manager.get_collector()
        is_done = await collector.add_data(request.client_id, request.image_token_len)
        if not is_done:
            await collector.event.wait()
        print(f'result: {collector.result}')
        return SelectedValue(
            client_id=request.client_id,
            selected=collector.result[request.client_id],
        )

async def serve():
    server = aio.server()
    add_DatasetServiceServicer_to_server(
        DatasetServiceImpl(world_size=3), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()

def start_server():
    asyncio.run(serve())


class DatasetServer:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self._server: Optional[aio.Server] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, port: int = 50051) -> None:
        def run_server():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            self._server = aio.server()
            add_DatasetServiceServicer_to_server(
                DatasetServiceImpl(world_size=self.world_size), self._server
            )
            self._server.add_insecure_port(f'[::]:{port}')
            
            self._loop.run_until_complete(self._server.start())
            self._loop.run_forever()

        self._thread = threading.Thread(
            target=run_server, 
            daemon=True, 
            name="gRPC Server Thread"
        )
        self._thread.start()

    def stop(self):
        asyncio.run_coroutine_threadsafe(self._stop(), self._loop).result()

    async def _stop(self) -> None:
        if self._server:
            await self._server.stop(grace=5)
        if self._loop:
            self._loop.stop()
