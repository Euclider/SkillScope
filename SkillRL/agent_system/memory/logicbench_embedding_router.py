"""Question-only Qwen embedding retrieval for the frozen SRA LogicBench bank."""

from .skillrl_embedding_batch_router import BatchedEmbeddingStepRouter
from .skillrl_embedding_router import EmbeddingRouterError
from .sra_logicbench_bank import SRA_LOGICBENCH19_MANIFEST_SHA256


class LogicBenchEmbeddingRouter(BatchedEmbeddingStepRouter):
    version = "skillscope-logicbench-question-batch-top1-v1"
    backend = "skillrl_embedding_logicbench_question_batch"
    query_formatter = "logicbench-question-only-json-v1"

    def _validate_memory(self, memory):
        if memory.bank.manifest_sha256 != SRA_LOGICBENCH19_MANIFEST_SHA256:
            raise EmbeddingRouterError("LogicBench routing requires the pinned 19-skill bank")

    def _visible_input(self, *, question: str):
        if not isinstance(question, str) or not question.strip():
            raise EmbeddingRouterError("LogicBench question must be nonempty text")
        return {"question": question}

    def route_question(self, question: str):
        return self.route_many([{
            "candidate_bundle": self.memory.retrieve(""),
            "question": question,
        }])[0]
