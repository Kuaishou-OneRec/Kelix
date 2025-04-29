import grpc
from dataset_pb2 import BatchSequenceInfo
from dataset_pb2_grpc import DatasetServiceStub
import argparse
import random
import asyncio
from typing import Any, Dict, List, Optional

async def client_async(client_id: int):
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = DatasetServiceStub(channel)
        arrays = [random.randint(100, 120) for _ in range(10)]
        response = await stub.BalanceSequence(BatchSequenceInfo(client_id=client_id, image_token_len=arrays))
        print(response.client_id, arrays, response.selected)

async def test():
    await asyncio.gather(
        client_async(0),
        client_async(1),
        client_async(2),
    )
    
    await asyncio.gather(
        client_async(1),
        client_async(2),
        client_async(0)
    )

## asyncio.run(test())


def balance_sequence(
    client_id: str,
    value: int,
    server_addr: str = "localhost:50051",
    timeout: float = 600.0
) -> dict:
    result_template = {
        "success": False,
        "result": None,
        "error": None
    }
    
    try:
        with grpc.insecure_channel(server_addr) as channel:
            stub = DatasetServiceStub(channel)
            response = stub.BalanceSequence(
                BatchSequenceInfo(client_id=client_id, image_token_len=value),
                timeout=timeout
            )
            
            result_template["success"] = True
            result_template["result"] = response.selected
            return result_template
            
    except grpc.RpcError as e:
        result_template["error"] = f"RPC Error [{e.code()}]: {e.details()}"
    except Exception as e:
        result_template["error"] = f"System Error: {str(e)}"
    
    return result_template
