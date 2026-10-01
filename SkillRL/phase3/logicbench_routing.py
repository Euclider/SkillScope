"""Question-only retrieval over evolving LogicBench snapshots, including GPU service."""
from agent_system.memory.logicbench_embedding_router import LogicBenchEmbeddingRouter
from agent_system.memory.sra_logicbench_bank import SRA_LOGICBENCH19_MANIFEST_SHA256

from .bank import Bank
from .common import require
from .embedding_routing import BranchBatchedEmbeddingRouter, BranchCache, EmbeddingRouterPool
from .routing import BranchMemory


class BranchLogicBenchRouter(BranchBatchedEmbeddingRouter):
    backend = "phase3_logicbench_question_batch"
    query_formatter = "logicbench-question-only-json-v1"
    _visible_input = LogicBenchEmbeddingRouter._visible_input
    route_question = LogicBenchEmbeddingRouter.route_question

    def _execution_version(self, execution):
        return "phase3-logicbench-question-batch-top1-v1"

    def _validate_memory(self, memory):
        require(isinstance(memory, BranchMemory) and isinstance(memory.bank, Bank)
                and memory.bank.source == SRA_LOGICBENCH19_MANIFEST_SHA256,
                "Expected a LogicBench-derived branch snapshot")

class LogicBenchRouterPool(EmbeddingRouterPool):
    """Same model/documents as Phase12; one active-bank index and cache per version."""
    def __init__(self, *args, **kwargs):
        # Keep this ledger separate from any ALF pool even when encoder is shared.
        super().__init__(*args, **kwargs)
        require(self.execution is not None, "LogicBench requires explicit batched encoder execution")

    def for_bank(self, bank):
        require(bank.branch_id == self.branch_id, "Foreign branch router")
        key = bank.manifest_sha256
        if key not in self.routers:
            path = self.ledger.path.with_suffix(".logicbench-banks") / f"{key}.sqlite3"
            router = BranchLogicBenchRouter(BranchMemory(bank), self.config,
                model_files=self.files, model_path=self.settings["model_path"],
                device=self.settings["device"], cache_path=path,
                max_local_calls=self.settings["max_local_calls"], execution=self.execution,
                _encoder=self.encoder)
            router.cache = BranchCache(path, router.protocol, self.settings["max_local_calls"], self.ledger)
            self.routers[key] = router
        return self.routers[key]
