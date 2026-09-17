# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .memory import SearchMemory, SimpleMemory
from .skills_only_memory import SkillsOnlyMemory
from .step_skill_router import FrozenStepSkillRouter

__all__ = [
    "SimpleMemory", "SearchMemory", "SkillsOnlyMemory", "FrozenStepSkillRouter",
    "RetrievalMemory", "SkillUpdater",
]


def __getattr__(name):
    """Do not import embedding/LLM dependencies on the template-only path."""
    if name == "RetrievalMemory":
        from .retrieval_memory import RetrievalMemory

        return RetrievalMemory
    if name == "SkillUpdater":
        from .skill_updater import SkillUpdater

        return SkillUpdater
    raise AttributeError(name)
