"""Data structures of the deliberation: typed claims, peer reactions, and the
verified trace that ultimately becomes the sample representation.

A *claim* is the atomic unit of evidence an agent puts forward. Typing it
(subject / relation / object drawn from controlled vocabularies) is what lets a
knowledge prior verify it formally — an untyped, free-form claim cannot be
looked up. The ablation `typed_claims: false` removes this structure on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ReactionType(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    EXTEND = "extend"
    ABSTAIN = "abstain"


class ClaimStatus(str, Enum):
    CONSENSUS = "consensus"     # supported by peers, no contradiction
    CONTESTED = "contested"     # at least one contradiction -> needs arbitration
    ORPHAN = "orphan"           # no active reaction -> discarded
    VERIFIED = "verified"       # survived the full protocol


@dataclass
class Claim:
    """A typed evidence statement: ``<subject> <relation> <object>``.

    e.g. ``"worst area" elevated_in "malignant"`` asserted with confidence 0.8.
    """
    subject: str                 # feature name (a KG-groundable entity)
    relation: str                # controlled relation, e.g. "elevated_in"
    object: str                  # class name (the hypothesis the claim supports)
    confidence: float            # agent self-confidence in [0, 1]
    modality: str                # which modality/agent produced it
    evidence: float              # raw signal (standardized feature value)
    grade: Optional[str] = None  # arbiter evidence grade E1..E4 (set in Round 3)
    status: ClaimStatus = ClaimStatus.ORPHAN

    def key(self) -> str:
        """Identity used to deduplicate / address a claim across rounds."""
        return f"{self.subject}|{self.relation}|{self.object}"

    def to_text(self) -> str:
        """Natural-language rendering — the auditable face of the representation."""
        verb = self.relation.replace("_", " ")
        grade = f" [{self.grade}]" if self.grade else ""
        return (f"{self.modality}: '{self.subject}' {verb} '{self.object}' "
                f"(conf={self.confidence:.2f}){grade}")


@dataclass
class Reaction:
    """A peer agent's response to another agent's claim during cross-examination."""
    claim_key: str
    reactor_modality: str
    type: ReactionType
    confidence: float


@dataclass
class DeliberationTrace:
    """The full record of one sample's deliberation.

    ``verified_claims`` is the representation substrate; the other fields are
    kept for interpretability, ablation, and the hallucination-containment metric.
    """
    sample_id: int
    all_claims: List[Claim] = field(default_factory=list)        # Round 1 output
    reactions: List[Reaction] = field(default_factory=list)      # Round 2 output
    verified_claims: List[Claim] = field(default_factory=list)   # Round 3 output
    pruned_claims: List[Claim] = field(default_factory=list)     # arbiter-rejected

    def reactions_for(self, claim: Claim) -> List[Reaction]:
        return [r for r in self.reactions if r.claim_key == claim.key()]

    def as_report(self) -> str:
        """Human-readable trace, used in qualitative figures and audits."""
        lines = [f"=== Deliberation trace (sample {self.sample_id}) ==="]
        lines.append(f"Round 1 — {len(self.all_claims)} candidate claims")
        lines.append(f"Round 2 — {len(self.reactions)} peer reactions")
        lines.append(f"Round 3 — {len(self.verified_claims)} verified claims:")
        for c in sorted(self.verified_claims, key=lambda x: -x.confidence):
            lines.append(f"   • {c.to_text()}")
        return "\n".join(lines)
