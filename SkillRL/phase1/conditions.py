from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
from typing import ContextManager


class SkillCondition(str, Enum):
    FULL_BANK = "full_bank"
    MINUS_SKILL = "minus_skill"
    NO_SKILL = "no_skill"


def apply_skill_condition(memory, condition: SkillCondition, skill_id: str | None) -> ContextManager[None]:
    """Return a non-mutating context for one of the three evaluation arms."""
    condition = SkillCondition(condition)
    if condition is SkillCondition.FULL_BANK:
        if memory.disabled_skill_ids:
            raise ValueError("FULL_BANK requires an empty pre-existing skill mask")
        return nullcontext()
    if condition is SkillCondition.MINUS_SKILL:
        if not skill_id:
            raise ValueError("MINUS_SKILL requires --skill-id")
        return memory.temporarily_disabled(skill_id)
    return memory.temporarily_disabled_all()

