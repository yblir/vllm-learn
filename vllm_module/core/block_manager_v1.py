"""A block manager that manages token blocks."""
import math
from abc import ABC, abstractmethod
from itertools import count, takewhile
from os.path import commonprefix
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple

from vllm_module.block import BlockTable, PhysicalTokenBlock
from vllm_module.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm_module.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from vllm_module.core.interfaces import AllocStatus, BlockSpaceManager
from vllm_module.logger import init_logger
from vllm_module.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm_module.utils import Device

logger = init_logger(__name__)


class BlockAllocatorBase(ABC):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    @abstractmethod
    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        pass

    @abstractmethod
    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def free(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_total_blocks(self) -> int:
        pass

    @abstractmethod
    def contains_block(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        pass


class CachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.current_num_blocks = 0
        self.cached_blocks: Dict[int, PhysicalTokenBlock] = {}

        self.evictor: Evictor = make_evictor(eviction_policy)

        self.default_hash_ctr = count()

    def allocate_block(self, block_hash: int,
                       num_hashed_tokens: int) -> PhysicalTokenBlock:
        if self.current_num_blocks == self.num_blocks:
            block = self.evictor.evict()
            block.block_hash = block_hash
            block.num_hashed_tokens = num_hashed_tokens
            return block
        block = PhysicalTokenBlock(device=self.device,
                                   block_number=self.current_num_blocks,
                                   block_size=self.block_size,
                                   block_hash=block_hash,
                                   num_hashed_tokens=num_hashed_tokens)
        self.current_num_blocks += 1
        return block

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if block_hash is None:
            block_hash = next(self.default_hash_ctr)
        if block_hash in self.evictor:
            assert block_hash not in self.cached_blocks
            block = self.evictor.remove(block_hash)
            assert block.ref_count == 0
            self.cached_blocks[block_hash] = block
            block.ref_count += 1
            assert block.block_hash == block_hash
            return block
        if block_hash not in self.cached_blocks:
            self.cached_blocks[block_hash] = self.allocate_block(
                    block_hash, num_hashed_tokens)
        block = self.cached_blocks[block_hash]
        assert block.block_hash == block_hash
        block.ref_count += 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            assert block.block_hash not in self.evictor
            self.evictor.add(block)

            # Remove the block from the cached_blocks
            del self.cached_blocks[block.block_hash]

    def get_num_free_blocks(self) -> int:
        return self.num_blocks - self.current_num_blocks + self.evictor.num_blocks

    def get_num_total_blocks(self) -> int:
        return self.num_blocks

    def contains_block(self, block_hash: int) -> bool:
        return block_hash in self.cached_blocks or block_hash in self.evictor

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        # Update the hash of block and the cached_blocks dictionary.
        assert not self.contains_block(block_hash)
        old_hash = block.block_hash
        block.block_hash = block_hash
        del self.cached_blocks[old_hash]
        self.cached_blocks[block_hash] = block


class UncachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
            self,
            device: Device,
            block_size: int,
            num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: BlockTable = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size,
                                       block_hash=-1,
                                       num_hashed_tokens=0)
            self.free_blocks.append(block)

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        """分配block: 从自由态block列表中取出一个block，并将引用计数设为1"""
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        """释放block，引用计数置为0"""
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        """获得当前gpu上可用block数量"""
        return len(self.free_blocks)

    def get_num_total_blocks(self) -> int:
        """获得当前gpu所有block总数"""
        return self.num_blocks

    def contains_block(self, block_hash: int) -> bool:
        raise NotImplementedError("Invalid codepath for uncached block allocator.")

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        raise NotImplementedError("Invalid codepath for uncached block allocator.")


class BlockSpaceManagerV1(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
            self,
            block_size: int,
            num_gpu_blocks: int,
            num_cpu_blocks: int,
            watermark: float = 0.01,
            sliding_window: Optional[int] = None,
            enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        if enable_caching and sliding_window is not None:
            raise NotImplementedError(
                    "Sliding window is not allowed with prefix caching enabled!")

        self.block_sliding_window = None
        if sliding_window is not None:
            # Round up to nearest block size to regularize sliding window
            # allocation sizes.
            self.block_sliding_window = math.ceil(sliding_window / block_size)

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching
        # 水位线，是一个数量阈值，设置它的目的是避免gpu上物理块全部使用完。
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # 根据是否做了prefix caching限制，来选择不同的allocator
        if self.enable_caching:
            logger.info("Automatic prefix caching is enabled.")
            self.gpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                    Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                    Device.CPU, block_size, num_cpu_blocks)
        else:
            self.gpu_allocator = UncachedBlockAllocator(
                    Device.GPU, block_size, num_gpu_blocks)
            self.cpu_allocator = UncachedBlockAllocator(
                    Device.CPU, block_size, num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        # 记录每个seq对应的BlockTable(这是一个包含物理块索引号的list)
        self.block_tables: Dict[int, BlockTable] = {}
        # Mapping: req_id -> BlockTable
        # Note that each SequenceGroup has a unique
        # request ID
        # 功能同上，但cross_block_tables记录的是encoder-decode类型的模型，暂时混略
        self.cross_block_tables: Dict[str, BlockTable] = {}

    def _get_seq_num_required_blocks(self, seq: Sequence) -> int:
        return 0 if seq is None else seq.n_blocks

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        # 只对encoder-decode模型有效，忽略
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # 计算当前seq序列需要的物理block数量
        # 这是seq的一个属性，对于waiting状态的seq，n_blocks=len(prompt)/16, 向上取整
        self_num_required_blocks = self._get_seq_num_required_blocks(
                seq_group.get_seqs(status=SequenceStatus.WAITING)[0])
        # 又是encoder-decode相关，忽略
        cross_num_required_blocks = self._get_seq_num_required_blocks(seq_group.get_encoder_seq())
        num_required_blocks = self_num_required_blocks + cross_num_required_blocks

        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks, self.block_sliding_window)
        # 当前gpu空闲的blocks数量
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        # 如果设备中所有的物理块数量 - 该seq实际需要的物理块数量 < 水位线block数量，则不分配
        # 说明当前seq太长了，标记为NEVER，以后也不处理这个seq_group了
        if self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks:
            return AllocStatus.NEVER
        # 如果设备中可用的物理块数量 - 该seq实际需要的block数量 >= 水位线block数量，则分配
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        # 否则，现在不能分配(暂时没足够的blocks)，但可以延迟分配
        else:
            return AllocStatus.LATER

    def _allocate_sequence(self, \
                           seq: Sequence, \
                           ref_count: int, \
                           is_encoder_decoder: bool = True) -> BlockTable:
        # Allocate new physical token blocks that will store the prompt tokens.
        # 当前seq需要的物理块数量
        num_prompt_blocks = seq.n_blocks

        block_table: BlockTable = []
        for logical_idx in range(num_prompt_blocks):
            # 滑窗，忽略
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
                # Set the reference counts of the token blocks.
                block.ref_count = ref_count
            elif not is_encoder_decoder and self.enable_caching:
                block = self.gpu_allocator.allocate(
                        seq.hash_of_block(logical_idx),
                        seq.num_hashed_tokens_of_block(logical_idx))
            # 默认情况下走下面的分支
            else:
                block = self.gpu_allocator.allocate()
                # Set the reference counts of the token blocks.
                # 由于seq_group下的所有seq共享一个prompt，所以有ref_count = num_seqs
                # 表示这些seqs的逻辑块都引用它了
                block.ref_count = ref_count
            block_table.append(block)

        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:
        is_encoder_decoder = seq_group.is_encoder_decoder()
        # 只对encoder-decode模型有效，忽略
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        # 对于WAITING装的seq_group，seq只有1条，就是prompt
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        # block_table:list,存储的是当前seq用到的物理块的索引号
        block_table: BlockTable = self._allocate_sequence(seq,
                                                          seq_group.num_seqs(),
                                                          is_encoder_decoder)

        # Assign the self-attention block tables for each sequence.
        # 记录每一个seq序列使用的block_table，block_tables是一个全局变量，记录这所有
        # seq_group的seq，根据add_request()中代码可知，不同seq_group的seq.id也不会重复，没有相互覆盖的风险
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            self.block_tables[seq.seq_id] = block_table.copy()

        # Allocate encoder sequence
        # 忽略
        if is_encoder_decoder:
            # A SequenceGroup has only a single encoder sequence (at most),
            # thus allocate with a ref count of 1
            block_table = self._allocate_sequence(seq_group.get_encoder_seq(),
                                                  1, is_encoder_decoder)
            # Assign the cross-attention block table for the SequenceGroup.
            self.cross_block_tables[seq_group.request_id] = block_table

    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerV1"

        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def _promote_last_block(
            self,
            seq: Sequence,
            last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        assert self.enable_caching

        # Compute a new hash for the block so that it can be shared by other
        # Sequences
        new_hash = seq.hash_of_block(seq.n_blocks - 1)

        # if new_hash is already in the cached table, then free last_block
        # and return the cached version
        if self.gpu_allocator.contains_block(new_hash):
            self.gpu_allocator.free(last_block)
            return self.gpu_allocator.allocate(new_hash)
        else:
            self.gpu_allocator.update_hash(new_hash, last_block)
            return last_block

    def _is_last_block_full(
            self,
            seq: Sequence,
    ) -> bool:
        token_ids_len = seq.data.get_len()
        return token_ids_len > 0 and token_ids_len % seq.block_size == 0

    def _maybe_promote_last_block(
            self,
            seq: Sequence,
            last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        if self._is_last_block_full(seq):
            return self._promote_last_block(seq, last_block)
        else:
            return last_block

    def _allocate_last_physical_block(
            self,
            seq: Sequence,
    ) -> PhysicalTokenBlock:
        # Called before a new block is appended.
        # This is in charge of allocating a new physical block (to be appended).

        # None if the last block is not full. Otherwise, we set it to the
        # content hash.
        if not self.enable_caching:
            return self.gpu_allocator.allocate()
        block_hash: Optional[int] = None
        n_blocks = seq.n_blocks
        if self._is_last_block_full(seq):
            block_hash = seq.hash_of_block(n_blocks - 1)
        num_hashed_tokens = seq.num_hashed_tokens_of_block(n_blocks - 1)

        # num_hashed_tokens is used to compute future hashes
        # (e.g. in the hashing function, it is used to ask the sequence for
        # prefix tokens)
        new_block = self.gpu_allocator.allocate(block_hash, num_hashed_tokens)

        # If the block has is None, then the block is not full.
        # If the block is not full, then we expect it to have a refcount of 1.
        if block_hash is None:
            assert new_block.ref_count == 1
        return new_block

    def append_slots(
            self,
            seq: Sequence,
            num_lookahead_slots: int = 0,
    ) -> List[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        n_blocks = seq.n_blocks
        # 读取这个seq的物理块，List[PhysicalTokenBlock]
        block_table = self.block_tables[seq.seq_id]
        # If we need to allocate a new physical block
        # 如果实际物理块数量 < seq需要的物理块数量(说明此时需要分配新的物理块了),为什么会出现这种情况?
        # 因为上1个推理阶段完毕后，seq的需求的块数量更新了，但物理块数量还没更新
        if len(block_table) < n_blocks:
            # Currently this code only supports adding one physical block
            # 需要声明物理块只允许比需求的块少1块
            assert len(block_table) == n_blocks - 1
            # 如果使用滑动窗口,忽略
            if self.block_sliding_window and len(block_table) >= self.block_sliding_window:
                # reuse a block
                block_table.append(block_table[len(block_table) % self.block_sliding_window])
            # 其余情况，直接分配一个新的物理块给当前序列
            else:
                # The sequence hash a new logical block.
                # Allocate a new physical block.
                new_block = self._allocate_last_physical_block(seq)
                block_table.append(new_block)
                return []

        # We want to append the token to the last physical block.
        # 取出最后一个物理块
        last_block = block_table[-1]
        # 断言该块必须是gpu物理块
        assert last_block.device == Device.GPU

        # 如果最后一个物理块的引用数量为1, 说明只有当前这个seq在用它
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            # 是在做prefix caching，暂时忽略
            if self.enable_caching:
                # If the last block is now complete, we may reuse an old block
                # to save memory.
                maybe_new_block = self._maybe_promote_last_block(seq, last_block)
                block_table[-1] = maybe_new_block
            return []
        # 如果最后一个物理块的引用数量为 > 1, 说明有别的seq在用它，不允许这样情况发生
        # 因为两个seq生成的内容可能不同，同时向一个位置添加kv-cache会出现相互覆盖的情况
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            # 触发copy-on-write机制，分配一个新的物理块
            new_block = self._allocate_last_physical_block(seq)
            # 用新分配的block替换之前分配的那个
            block_table[-1] = new_block
            # 把之前分配的block释放掉, 也即该物理块ref_count -= 1，
            # 如果-=1后ref_count=0，说明该物理块变为自由状态；但当前语境下不可能为0，因为
            # 正是因为last_block.ref_count>1才会走到这里，此时last_block.ref_count最小为1
            self.gpu_allocator.free(last_block)
            return [(last_block.block_number, new_block.block_number)]

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        # When using a sliding window, blocks will be eventually reused.
        # In this case the block tables will contain repeated blocks.
        # When forking, we must make sure that each block's `ref_count`
        # is only incremented by one, so we deduplicate them by wrapping
        # them in a set.
        for block in set(src_block_table):
            block.ref_count += 1

    def _get_physical_blocks(self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:

        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        request_id = seq_group.request_id
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        # Cross-attention blocks
        if seq_group.is_encoder_decoder():
            blocks.update(self.cross_block_tables[request_id])
        return list(blocks)

    def can_swap_in(self,
                    seq_group: SequenceGroup,
                    num_lookahead_slots: int = 0) -> AllocStatus:
        assert num_lookahead_slots == 0, "BlockSpaceManagerV1 does not support lookahead allocation"
        # 当前seq_group正在使用的不重复的物理块
        blocks = self._get_physical_blocks(seq_group)
        # 当前处于SWAPPED状态的seq数量
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        # 忽略
        if seq_group.is_encoder_decoder():
            num_swapped_seqs += 1
        # 当前GPU可用的物理块数量
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().

        # len(blocks)是移动回GPU时应该使用的物理块数量，prompt+已完成解码的output 的kv-cache 需要使用这些block
        # num_swapped_seqs是预备生成的token所使用的block，前面我们分析过，解码阶段，一个seq可能使用的
        # block最小为0(最后一个block槽位没满，还能继续添加)，最大为1(最后的block槽位满，要新增block才能完成推理)
        # 随意二者加起来的block的数量才是能绝对满足该seq_group推理的block数量
        num_required_blocks = len(blocks) + num_swapped_seqs
        # 如果GPU总共的blocks(不是可用block，是所有的block)都小于num_required_blocks，
        # 这条seq_group没法推理(GPU装不下这条数据)，
        if self.gpu_allocator.get_num_total_blocks() < num_required_blocks:
            return AllocStatus.NEVER
        # 在水位线以上，合格
        elif num_free_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        # 小于水位线，GPU block数量暂时不够，稍后在处理这条数据
        else:
            return AllocStatus.LATER

    def _swap_block_table(
            self, block_table: BlockTable, src_allocator: BlockAllocatorBase,
            dest_allocator: BlockAllocatorBase,
            mapping: Dict[PhysicalTokenBlock,
            PhysicalTokenBlock]) -> BlockTable:
        new_block_table = []

        for from_block in block_table:
            # mapping 为空，走不到if
            if from_block in mapping:
                to_block = mapping[from_block]
                to_block.ref_count += 1
            else:
                # 在CPU上分配物理块
                to_block = dest_allocator.allocate(
                        from_block.block_hash, from_block.num_hashed_tokens)
                # 记录GPU与CPU上物理块的索引号映射，便于以后cpu->gpu找回。
                mapping[from_block] = to_block
            # 记录CPU物理块的索引号，CPU物理块与CPU物理块一一对应
            new_block_table.append(to_block)
            # Free the source block swapped in to destination.
            # 释放GPU物理块
            src_allocator.free(from_block)

        return new_block_table

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:

        request_id = seq_group.request_id

        # CPU block -> GPU block.
        # dict is efficient in lookup `if cpu_block in mapping`
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            self.block_tables[seq.seq_id] = \
                self._swap_block_table(self.block_tables[seq.seq_id],   # 取出该seq用到的GPU block
                                       self.cpu_allocator,  # CPU物理块分配器
                                       self.gpu_allocator,  # GPU物理块分配器
                                       mapping)

        if seq_group.is_encoder_decoder():
            self.cross_block_tables[request_id] = \
                self._swap_block_table(self.cross_block_tables[request_id],
                                       self.cpu_allocator,
                                       self.gpu_allocator,
                                       mapping)

        return [(cpu_block.block_number, gpu_block.block_number)
                for cpu_block, gpu_block in mapping.items()]

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        request_id = seq_group.request_id

        # GPU block -> CPU block.
        # dict is efficient in lookup `if gpu_block in mapping`
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        # 遍历当前seq_group中每条seq，gpu->cpu
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            self.block_tables[seq.seq_id] = \
                self._swap_block_table(self.block_tables[seq.seq_id],
                                       self.gpu_allocator,
                                       self.cpu_allocator,
                                       mapping)
        # 忽略
        if seq_group.is_encoder_decoder():
            self.cross_block_tables[request_id] = \
                self._swap_block_table(self.cross_block_tables[request_id],
                                       self.gpu_allocator,
                                       self.cpu_allocator,
                                       mapping)

        return [(cpu_block.block_number, gpu_block.block_number)
                for cpu_block, gpu_block in mapping.items()]

    def _free_block_table(self, block_table: BlockTable) -> None:
        # when using a sliding window, each seq will only use up
        # to `self.block_sliding_window` blocks. When freeing
        # the block table, we must make sure to not free blocks more
        # than once. If no sliding window is used, there is no block
        # reuse in the block table, so we must free all blocks.
        blocks_to_free = (block_table[-self.block_sliding_window:]
                          if self.block_sliding_window is not None else
                          block_table)
        for block in set(blocks_to_free):
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def free_cross(self, seq_group: SequenceGroup) -> None:
        if seq_group.request_id not in self.cross_block_tables:
            # Already freed or hasn't ben scheduled yet.
            return
        block_table = self.cross_block_tables[seq_group.request_id]
        self._free_block_table(block_table)
        del self.cross_block_tables[seq_group.request_id]

    def reset(self) -> None:
        # Free decoder block tables
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()
        # Free cross-attention block tables
        for block_table in self.cross_block_tables.values():
            self._free_block_table(block_table)
        self.cross_block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        block_table = self.cross_block_tables[seq_group.request_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()

    def access_all_blocks_in_seq(
            self,
            seq: Sequence,
            access_time: float,
    ) -> None:
        if self.enable_caching:
            # Update the last accessed time of all the blocks accessed
            # in this step.
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                block.last_accessed = access_time

    def compute_full_blocks_in_seq(self, seq: Sequence):
        if seq.seq_id not in self.block_tables:
            return
        max_full_block = seq.get_len() // self.block_size - 1
        block_table = self.block_tables[seq.seq_id]
        if max_full_block == -1:
            return
        for i in reversed(range(max_full_block)):
            if block_table[i].computed:
                break
            block_table[i].computed = True

    def get_all_computed_blocks(self, seq: Sequence) -> List[int]:
        if seq.seq_id not in self.block_tables:
            return []
        block_table = self.block_tables[seq.seq_id]
        # NOTE We exclude the last block to avoid the case where the entire
        # prompt is cached. This would cause erroneous behavior in model
        # runner.
        return [b.block_number for b in takewhile(lambda b: b.computed, block_table[:-1])]

    def get_common_computed_block_ids(self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Return the block ids that are common for a given sequence group.

        Used in prefill (can skip prefill of some blocks).
        """
        # Can return non-empty result only with prefix caching enabled.
        if not self.enable_caching:
            return []

        ids_list = [self.get_all_computed_blocks(seq) for seq in seqs]
        return commonprefix([ids for ids in ids_list if ids != []])

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        if self.enable_caching:
            for seq in seq_group.get_seqs():
                self.compute_full_blocks_in_seq(seq)
