"""OmicsCouncil: a domain-agnostic deliberative multi-agent framework that
represents a sample by the *trace of an agent deliberation* grounded in a
knowledge prior, rather than by an opaque latent vector.

The public surface is intentionally small; see ``pipeline.OmicsCouncilPipeline``
for the end-to-end entry point used by the experiment scripts.
"""

from .config import Config, load_config
from .claims import Claim, Reaction, DeliberationTrace, ClaimStatus, ReactionType
from .pipeline import OmicsCouncilPipeline

__all__ = [
    "Config",
    "load_config",
    "Claim",
    "Reaction",
    "DeliberationTrace",
    "ClaimStatus",
    "ReactionType",
    "OmicsCouncilPipeline",
]

__version__ = "0.1.0"
