"""Token blocks."""
from typing import List

from vllm_module.utils import Device

DEFAULT_LAST_ACCESSED_TIME = -1


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
    ) -> None:
        self.device = device
        # 该物理块在对应设备上的全局block索引号
        self.block_number = block_number
        # 每个block槽位数量(默认16)
        self.block_size = block_size
        # 在prefix caching场景下使用，其他场景值为-1
        self.block_hash = block_hash
        # 该物理块的hash值是由多少个前置token计算而来的，非prefix caching场景值为0
        self.num_hashed_tokens = num_hashed_tokens
        # 该物理块被引用次数
        self.ref_count = 0
        # 物理块最后一个被访问时间，非prefix caching场景值为-1
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME
        # 该物理块是否被计算过，只在prefix caching场景下启用
        self.computed = False

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'computed={self.computed})')


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]
