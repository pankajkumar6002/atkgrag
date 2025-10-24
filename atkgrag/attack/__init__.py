"""Attack module initialization."""

from .query_attack import QueryLevelAttack
from .prompt_attack import PromptBasedAttack

__all__ = ["QueryLevelAttack", "PromptBasedAttack"]
