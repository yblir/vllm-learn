import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm_module.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm_module.core.interfaces import AllocStatus, BlockSpaceManager
from vllm_module.logger import init_logger
from vllm_module.lora.request import LoRARequest
from vllm_module.prompt_adapter.request import PromptAdapterRequest
from vllm_module.sequence import (Sequence, SequenceData, SequenceGroup,
                                  SequenceGroupMetadata, SequenceStatus)

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
        os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            return

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_seq_groups: Iterable[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        self.scheduled_seq_groups = sorted(
                self.scheduled_seq_groups,
                key=lambda g: (g.seq_group.lora_int_id, g.seq_group.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
                decode_seq_groups=[],
                prefill_seq_groups=[],
                preempted=[],
                swapped_out=[],
                blocks_to_swap_out=[],
                blocks_to_copy=[],
                num_lookahead_slots=0,
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
                decode_seq_groups=[],
                prefill_seq_groups=[],
                blocks_to_swap_in=[],
                blocks_to_copy=[],
                num_lookahead_slots=0,
                infeasible_seq_groups=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    # Selected sequences for prefill.
    seq_groups: List[SequenceGroup]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
                seq_groups=[],
                ignored_seq_groups=[],
                num_lookahead_slots=0,
        )


class Scheduler:

    def __init__(
            self,
            scheduler_config: SchedulerConfig,
            cache_config: CacheConfig,
            lora_config: Optional[LoRAConfig],
            pipeline_parallel_size: int = 1,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "v1"
        if self.scheduler_config.use_v2_block_manager:
            version = "v2"
        if self.scheduler_config.embedding_mode:
            version = "embedding"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
                version)

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
                block_size=self.cache_config.block_size,
                num_gpu_blocks=num_gpu_blocks,
                num_cpu_blocks=num_cpu_blocks,
                sliding_window=self.cache_config.sliding_window,
                enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()
        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        # This is used to evict the finished requests from the Mamba cache.
        self._finished_requests_ids: List[str] = list()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        # preemption mode, RECOMPUTE or SWAP
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the running queue.
        # Only for testing purposes.
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the swapped queue.
        # Only for testing purposes.
        self.swapped.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                # Remove the aborted request from the Mamba cache.
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(self.swapped) != 0

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _schedule_running(
            self,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        # todo 类型变了
        # blocks_to_swap_out：{gpu物理块id: cpu物理块id}
        # blocks_to_copy: {旧物理块id：[由旧物理块copy-on-write而来的新物理块id]}
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.

        running_queue = self.running

        while running_queue:
            seq_group = running_queue[0]
            # 当前待推理的seq_group,需要处理,或者说准备返回的tokens数量
            num_running_tokens = self._get_num_new_tokens(
                    seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            # todo 觉得这个判断有点多余,因为处于RUNNING状态的seq,必定有tokens返回,num_running_tokens
            # todo 不可能为0, 再说,如果为0, 方法self._get_num_new_tokens内部就会抛出异常,因为做了assert断言
            if num_running_tokens == 0:
                break

            # 经过num_running_tokens检验没问题后, 将该seq_group从running_queue中取出来
            running_queue.popleft()

            # 对于这个seq_group，检查对于其中的每一个seq，是否能至少分配一个物理块给它，如果不能的话
            # （说明要执行抢占操作了，否则马上会没有资源让这个最早到达的seq_group做完推理）：
            # （注意，这里用了while...else，如果while条件正常结束，则进入else内容；
            #  如果被break，则不会执行else）
            while not self._can_append_slots(seq_group):  # 如果不能为每个seq都分配一个block
                # 这个seq_group本来是要送去做推理的,但没有足够的gpu物理blocks分配给它,对不住了,只能把它
                # 加入swap队列,以后有机会再捞它
                # 之前这个seq_group准备返回的tokens数量已经加到budget属性上,现在不处理它, 要把数量再减回来
                # todo 什么时候加的呀? 每次循环都要减一次?
                budget.subtract_num_batched_tokens(seq_group.request_id, num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                # seq也要减回来, 是在外层的self._schedule_default 加上的
                budget.subtract_num_seqs(seq_group.request_id, num_running_seqs)

                # lora相关,忽略
                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # ------------------------------------------------------------------------------------------------------
                # 经过以下释放gpu blocks工作后,再次进入while循环判断gpu blocks数量是否够用,
                # 如果够用,进入到与while对应的else分支,如果不够用,继续释放gpu blocks,直到够用或running_queue全部取完.
                # ------------------------------------------------------------------------------------------------------

                # 如果此时running_queue队列不为空,把最后一个seq_group踢出去放入swap队列,给
                # 上面这个seq_group腾位置(释放最后一个seq_group对应的gpu blocks)
                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()

                    # 有两种swap方式,RECOMPUTE:删除所有,回炉到waiting队列. SWAP:将blocks全部转移到CPU blocks上
                    preempted_mode = self._preempt(victim_seq_group, blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                # 如果running_queue队列已经空了,没有替罪的羊,只能把自己放入swap队列了.
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._preempt(seq_group, blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(seq_group)
                    else:
                        swapped_out.append(seq_group)
                    # 此时running_queue队列已空,已经没有seq_group可处理了,使用break中断
                    # while循环, 不走后面的else分支,直接return,而且本次调度没有指定任何待推理的seq_group
                    break
            else:
                # 为当前seq_group分配gpu 物理blocks. 这里只分配了逻辑blocks与物理blocks的映射关系
                # blocks_to_copy:[旧物理块id, copy - on - write而来的新物理块id]
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(ScheduledSequenceGroup(seq_group=seq_group,
                                                                     token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(ScheduledSequenceGroup(seq_group=seq_group,
                                                                    token_chunk_size=1))
                # todo 不明白与上面budget.subtract_num_batched_tokens的对应关系
                budget.add_num_batched_tokens(seq_group.request_id, num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                # 默认情况下, 以下两个if都走不到
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        return SchedulerRunningOutputs(
                decode_seq_groups=decode_seq_groups,
                prefill_seq_groups=prefill_seq_groups,
                preempted=preempted,
                swapped_out=swapped_out,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=False))

    def _schedule_swapped(
            self,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        # [(cpu物理块id, gpu物理块id)]
        blocks_to_swap_in: List[Tuple[int, int]] = []
        # [(旧物理块,copy - on - write而来的新物理块id)]
        blocks_to_copy: List[Tuple[int, int]] = []
        # 准备解码的seq_group
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        # 准备预填充的seq_group
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        # 因各种原因，被标记为不再处理的seq_group，如预填充序列太长了...
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            # 取出swap队列中最早被抢占的seq_group
            seq_group = swapped_queue[0]

            # ----------------------------------------------------------------------------------------------------------
            # If the sequence group cannot be swapped in, stop.
            # 对被抢占seq_group有两种处理方式，1. 清空放入waiting队列，这时is_prefill为True
            # 2.blocks全部转移到CPU上，这时is_prefill为False
            # self._get_num_lookahead_slots(is_prefill)必定为0，否则抛出异常，block_manager_v1不支持非0
            # 情况，todo 搞不懂设置这个值是干嘛的！
            is_prefill = seq_group.is_prefill()
            # 根据需要的，与可用的物理blocks数量判断，是否可以把当前seq_group从swap队列转移到running队列
            alloc_status = self.block_manager.can_swap_in(
                    seq_group, self._get_num_lookahead_slots(is_prefill))
            if alloc_status == AllocStatus.LATER:    # 稍后，资源多时再处理
                break
            elif alloc_status == AllocStatus.NEVER:  # 不合格，永不再处理
                logger.warning(
                        "Failing the request %s because there's not enough kv "
                        "cache blocks to run the entire sequence.",
                        seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue
            # ----------------------------------------------------------------------------------------------------------

            # lora相关，忽略
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            # 取出这个seq_group在剩余生命周期内将并行运行的最大seq数量
            num_new_seqs = seq_group.get_max_num_running_seqs()
            # 当前准备转移的seq_group,需要处理,或者说准备返回的tokens数量，
            # decode模式：每个seq num_token=1,其他模式则遵循各自的状态
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)
            # 感觉num_new_tokens==0的判断有点多余，基本不可能为0
            # budget.can_schedule 会判断加上当前seq_group的num_new_tokens和num_new_seqs后
            # 总数是否会超标，如果超标，说明不能再添加任何seq_group到running队列，直接结束本次调度
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens, num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)

            # 如果能走到这步，说明可向running队列转移了。先把当前seq_group从swap队列踢出来
            # 再把CPU上的blocks转移到GPU block上
            # todo 再append？
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            # 判断是不是预填充，将这个seq_group加入不同的分组
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(ScheduledSequenceGroup(seq_group, token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(ScheduledSequenceGroup(seq_group, token_chunk_size=1))

            # 将这个马上上岸的seq_group的tokens和seqs数量更新到budget中
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
                decode_seq_groups=decode_seq_groups,
                prefill_seq_groups=prefill_seq_groups,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_copy=blocks_to_copy,
                num_lookahead_slots=self._get_num_lookahead_slots(
                        is_prefill=False),
                infeasible_seq_groups=infeasible_seq_groups,
        )

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if (seq_group.lora_request
                and seq_group.lora_request.long_lora_max_len):
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _schedule_prefills(
            self,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # ignored_seq_groups：记录因太长（所需的blocks和总blocks之间的差值超过阈值了），
        # 而无法继续做生成的seq_group，这些seq_group中的seq状态都会被标记为
        # FINISHED_IGNORED，表示直接不处理他们
        ignored_seq_groups: List[SequenceGroup] = []
        # 用于装载从wait队列转移出来的seq_group
        seq_groups: List[SequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        # self._passed_delay：通过比较当前请求到达时间来确定是否要从wait队列拿任务
        # 因为拿任务也要占用时间，需要平衡调度任务与推理任务的调用时机
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
            # list,从当前seq_group取出准备推理的seq序列. 1个prompt可能有多个seq(多输出)，但wait队列中连预填充
            # 都没进行，因此这时的seq(仅是prompt)数量必定==1
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, "Waiting sequence group should have only one prompt sequence."

            # 获取当前seq的序列长度（如果该seq_group来自之前被抢占的请求，
            # 那么这个长度不仅包括prompt，还包括已经处理好的output token）
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            # 如果这条seq的长度 > 每次调度能处理的最大序列长度，那么把这条seq的状态置为FINISHED_IGNORED，
            # 并将对应seq_group装入ignored_seq_groups中，然后将其从waiting列表中移除，永不再处理，完结撒花~
            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                        "Input prompt (%d tokens) is too long"
                        " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)  # 加入失败者联盟
                waiting_queue.popleft()  # 从 waiting 队列中踢出去
                continue  # 继续从waiting拿数据处理

            # If the sequence group cannot be allocated, stop.
            # 比较当前seq需要的物理块,gpu可用物理块之间的数量关系. 决定是否能给当前seq_group分配物理块
            # can_allocate返回值可能有三种： NEVER：不分配；OK：可以分配；LATER：延迟分配
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            # 当前seq需要的blocks数量,超过gpu能提供的最大数量.加入失败者联盟,永不再处理，
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                        f"Input prompt ({num_new_tokens} tokens) is too long"
                        " and exceeds the capacity of block_manager")
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # lora相关，忽略
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            # 当前seq_group中状态为 未执行完 的序列的数量，即seq还没推理完成的数量. 刚从wait中取出时，
            # seq数量是1,但推理生成阶段,这个seq_group中会有n个seq在并行.n是外部传入的output数量. 因此这里num_new_seqs==n
            num_new_seqs = seq_group.get_max_num_running_seqs()
            # budget.can_schedule同时判断tokens和seqs数量是否超过阈值，任一个超过单次调度能执行的总数的阈值
            # 说明这step可推理的seqs数量已经马上趋于饱和，不能再加入seq到running队列。跳出while, 结束本次waiting向running的调度
            if num_new_tokens == 0 or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                              num_new_seqs=num_new_seqs):
                break

            # Can schedule this request.
            # lora相关，忽略
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)

            # 走到这一步时，说明当前seq_group已经通过上述种种验证，可以被加入running队列进行推理
            # 先将其从waiting队列中移出
            waiting_queue.popleft()
            # 为当前seq_group分配物理块,并将该seq_group中每条seq的status从waiting改为running
            self._allocate_and_set_running(seq_group)

            # ScheduledSequenceGroup类似于C++结构体。仅包含seq_group和token_chunk_size两个变量
            # 搞不懂vllm为什么总喜欢这种包裹操作，在各处代码中随处可见。用基本的list,或dict不好吗！
            seq_groups.append(
                    ScheduledSequenceGroup(seq_group=seq_group,
                                           token_chunk_size=num_new_tokens))

            # 当前seq_group的tokens和seqs数量增加到预算budget中
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        # 和lora相关的操作，忽略
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
                seq_groups=seq_groups,
                ignored_seq_groups=ignored_seq_groups,
                num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.

        当前策调度略旨在优化吞吐量。首先，它会批量处理尽可能多的预填充请求。然后它会安排解码。
        如果 GPU 内存有压力，则可以交换或抢占解码请求。因此会优先从swapped进行判断。
        """
        # Include running requests to the budget.
        # 每次step都要重新初始化一个budget来管理本次调度的的tokens和seqs数量, 根据数量是否超过阈值，决定将本次
        # seq_groups放入哪个队列。(一个seq_groups会包含多个seqs)
        budget = SchedulingBudget(
                token_budget=self.scheduler_config.max_num_batched_tokens,
                max_num_seqs=self.scheduler_config.max_num_seqs,
        )

        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        # 先统计正在执行推理的seq_groups中seq的数量
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id, seq_group.get_max_num_running_seqs())
        # lora推理相关，可忽略
        curr_loras = set(
                seq_group.lora_int_id for seq_group in self.running
                if seq_group.lora_int_id > 0) if self.lora_enabled else None

        # 以下三个变量，类似于C++中的结构体。将多个变量合在一起，通过.属性访问
        # 各自保存处于不同活跃状态(wait,run,swap)的seq_groups具有的属性
        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # If any requests are swapped, prioritized swapped requests.
        # 为什么要从swap开始判断？
        # 调度的任务是优化吞吐量，即保证处于running状态的seqs最多。running从wait和swap队列
        # 获得，首先积压的任务可能要比wait的优先级高，因为swap队列中的任务始终占据着系统资源，当
        # running可添加时，应该首先处理swap。
        # 为什么不先判断running是否饱和？ 老板当然要先把工作甩你脸上，难道要先为你工作是否饱和？不饱和就干，饱和了就再积压，
        # 如此才能最大化压榨打工人。
        if not self.swapped:  # 如果swapped队列为空
            # 既然不能从swap想running转移，那就只能从wait队列拿任务了。
            # wait队列中的都是原始任务，第一步要预填充
            # prefills是一个伪结构体：可以.出以下属性
            #     seq_groups: List[SequenceGroup]
            #     # Ignored sequence groups.
            #     ignored_seq_groups: List[SequenceGroup]
            #     num_lookahead_slots: int
            prefills = self._schedule_prefills(budget, curr_loras, enable_chunking=False)

        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.

        # self.waiting空,或 self.swapped非空,都会导致prefills.seq_groups数量为0
        # # 这个判断的意思是,prefills.seq_groups==0,说明本次调度没有安排预填充任务,那么就安排解码任务.
        # 执行推理任务的seq_group都在running队列，因此需要对这个队列进行调度。
        # 调度什么呢？
        # 是看running队列中的seq_group是否可以继续做推理任务。因为vllm动态管理，最大限度优化吞吐量，会导致blocks资源紧张
        # 上次推理生成的tokens的kv-cache需要GPU blocks去存储，导致资源消耗。那么这次准备推理时blocks数量不一定能够它完成
        # 推理，所以要对running队列中每个seq_group进行检查，看是否可以进行做推理。
        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(budget, curr_loras, enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            # 在对running队列调度后(从self.running队列取seq_group,准备进行推理),如果没有seq_group被
            # 抢占(退回wait队列),也没有seq_group被转移到CPU上, 说明blocks资源充足,可以把以前
            # self.swapped队列中积压的seq_group转移到gpu blocks做推理.

            # 注意这几个队列的判断逻辑. 如果self.swapped原本就非空,会进入上面的if判断分支进行self.running队列
            # 调度取值.然后根据这个过程中是否有seq_group被preempted和swapped_out获知blocks资源使用情况.
            # 如果没有被preempted和swapped_out.说明工作不饱和，就把self.swapped内容取出来，加入running队列进行推理
            # 如果有被preempted和swapped_out，说明资源紧张. self.swapped积压的任务暂时还处理不了
            # 如果下面if为False,意思是就不要再从self.swapped转移seq_group会gpu做推理了
            if len(running_scheduled.preempted) + len(running_scheduled.swapped_out) == 0:
                swapped_in = self._schedule_swapped(budget, curr_loras)

        # 最后一次判断本次推理的tokens和seqs数量是否超过阈值
        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        # 这个类型被抢占的seq_group，打回原型，重新加入waiting队列。幸运的是添加到了队列头部，当再次从
        # waiting队列取数据时，会优先处理它
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        # 将以上通过层层筛选的seq_group加入到running队列(真·running)，这些seq_group才是下一步的推理对象
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend([s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])
        # Update swapped requests.
        # 没有足够资源做推理的seq_group会从running转移到swap队列(swap队列是路径之一，另一个是加入到wait队列)
        self.swapped.extend(running_scheduled.swapped_out)
        # 统计被抢占的seq_group数量
        preempted = len(running_scheduled.preempted) + len(running_scheduled.swapped_out)

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        return SchedulerOutputs(
                scheduled_seq_groups=(prefills.seq_groups +
                                      running_scheduled.decode_seq_groups +
                                      swapped_in.decode_seq_groups),
                num_prefill_groups=len(prefills.seq_groups),
                num_batched_tokens=budget.num_batched_tokens,
                blocks_to_swap_in=swapped_in.blocks_to_swap_in,
                blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
                blocks_to_copy=running_scheduled.blocks_to_copy +
                               swapped_in.blocks_to_copy,
                ignored_seq_groups=prefills.ignored_seq_groups +
                                   swapped_in.infeasible_seq_groups,
                num_lookahead_slots=running_scheduled.num_lookahead_slots,
                running_queue_size=len(self.running),
                preempted=preempted,
        )

    def _schedule_chunked_prefill(self):
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
                token_budget=self.scheduler_config.max_num_batched_tokens,
                max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        prefills = SchedulerPrefillOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        running_scheduled = self._schedule_running(budget,
                                                   curr_loras,
                                                   enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget, curr_loras)

        # Schedule new prefills.
        prefills = self._schedule_prefills(budget,
                                           curr_loras,
                                           enable_chunking=True)

        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend([s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend([s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend([s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        return SchedulerOutputs(
                scheduled_seq_groups=(prefills.seq_groups +
                                      running_scheduled.prefill_seq_groups +
                                      swapped_in.prefill_seq_groups +
                                      running_scheduled.decode_seq_groups +
                                      swapped_in.decode_seq_groups),
                num_prefill_groups=(len(prefills.seq_groups) +
                                    len(swapped_in.prefill_seq_groups) +
                                    len(running_scheduled.prefill_seq_groups)),
                num_batched_tokens=budget.num_batched_tokens,
                blocks_to_swap_in=swapped_in.blocks_to_swap_in,
                blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
                blocks_to_copy=running_scheduled.blocks_to_copy +
                               swapped_in.blocks_to_copy,
                ignored_seq_groups=prefills.ignored_seq_groups +
                                   swapped_in.infeasible_seq_groups,
                num_lookahead_slots=running_scheduled.num_lookahead_slots,
                running_queue_size=len(self.running),
                preempted=(len(running_scheduled.preempted) +
                           len(running_scheduled.swapped_out)),
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        # Appending slots only occurs in decoding.
        is_prefill = False

        return self.block_manager.can_append_slots(
                seq_group=seq_group,
                num_lookahead_slots=self._get_num_lookahead_slots(is_prefill),
        )

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        # 该函数调用改变调度的内部状态(self.running、self.swapped 和 self.waiting)
        #
        scheduler_outputs = self._schedule()
        now = time.time()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            if seq_group.is_prefill():
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if token_chunk_size + seqs[0].data.get_num_computed_tokens() < seqs[0].data.get_len():
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            is_prompt = seq_group.is_prefill()
            seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=seq_group.multi_modal_data
                    if scheduler_outputs.num_prefill_groups > 0 else None,
                    prompt_adapter_request=seq_group.prompt_adapter_request,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(scheduled_seq_group.seq_group)

        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        remaining: Deque[SequenceGroup] = deque()
        for seq_group in self.running:
            if seq_group.is_finished():
                # Add the finished requests to the finished requests list.
                # This list will be used to update the Mamba cache in the
                # next step.
                self._finished_requests_ids.append(seq_group.request_id)
            else:
                remaining.append(seq_group)
        self.running = remaining

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
            self,
            seq_group: SequenceGroup,
            blocks_to_copy: List[Tuple[int, int]],
    ) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
        """
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False)

        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            blocks_to_copy.extend(cows)

    def _preempt(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: List[Tuple[int, int]],
            preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if self.user_specified_preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        elif self.user_specified_preemption_mode == "swap":
            preemption_mode = PreemptionMode.SWAP
        else:
            preemption_mode = PreemptionMode.RECOMPUTE

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                    "Sequence group %s is preempted by %s mode because there is "
                    "not enough KV cache space. This can affect the end-to-end "
                    "performance. Increase gpu_memory_utilization or "
                    "tensor_parallel_size to provide more KV cache memory. "
                    "total_num_cumulative_preemption=%d", seq_group.request_id,
                    preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt_by_recompute(
            self,
            seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()

    def _preempt_by_swap(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                    "Aborted due to the lack of CPU swap space. Please increase "
                    "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min([e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                    (now - earliest_arrival_time) >
                    (self.scheduler_config.delay_factor * self.last_prompt_latency)
                    or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.
        """
        if is_prefill:
            return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_tokens(self, seq_group: SequenceGroup,
                            status: SequenceStatus, enable_chunking: bool,
                            budget: SchedulingBudget) -> int:
        """Get the next new tokens to compute for a given sequence group
            that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns 0 if the new token cannot be computed due to token budget.
        """
        num_new_tokens = 0
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            num_new_tokens += seq.get_num_new_tokens()
        assert num_new_tokens > 0
        # Chunk if a running request cannot fit in.
        # If number of seq > 1, it means it is doing beam search in a
        # decode phase. Do not chunk in that case.
        if enable_chunking and len(seqs) == 1:
            num_new_tokens = min(num_new_tokens, budget.remaining_token_budget())
        return num_new_tokens
