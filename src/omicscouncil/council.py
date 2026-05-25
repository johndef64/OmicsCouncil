"""The deliberation protocol: the three rounds that turn raw modalities into a
verified deliberation trace.

  Round 1  independent typed claims          (agents, in isolation)
  Round 2  peer-to-peer cross-examination    (support / contradict / extend)
  Round 3  graph-mediated arbitration        (evidence grading + pruning)

Each round can be switched off via :class:`~omicscouncil.config.CouncilConfig`,
which is exactly how the ablations in ``experiments/run_ablation.py`` are run.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .agents import Agent
from .claims import Claim, ClaimStatus, DeliberationTrace, ReactionType
from .config import CouncilConfig
from .knowledge import KnowledgePrior


class Council:
    def __init__(self, agents: Dict[str, Agent], prior: KnowledgePrior, cfg: CouncilConfig):
        self.agents = agents
        self.prior = prior
        self.cfg = cfg

    def deliberate(self, rows: Dict[str, np.ndarray], sample_id: int = -1) -> DeliberationTrace:
        """Run the full protocol for one sample.

        ``rows`` maps modality name -> that modality's feature vector. A modality
        absent from ``rows`` is simply not represented: its agent does not speak,
        and the protocol proceeds with the rest — this is the structural answer
        to missing modalities (no imputation, no placeholder).
        """
        trace = DeliberationTrace(sample_id=sample_id)

        # --- Round 1: independent typed claims --------------------------------
        for mod, agent in self.agents.items():
            if mod not in rows:
                continue
            trace.all_claims.extend(agent.emit_claims(rows[mod]))

        # --- Round 2: peer cross-examination ----------------------------------
        surviving: List[Claim]
        if self.cfg.enable_cross_examination:
            surviving = []
            for claim in trace.all_claims:
                supports = contradicts = 0
                for mod, agent in self.agents.items():
                    if mod == claim.modality or mod not in rows:
                        continue
                    r = agent.react(claim, rows[mod])
                    trace.reactions.append(r)
                    if r.type in (ReactionType.SUPPORT, ReactionType.EXTEND):
                        supports += 1
                    elif r.type == ReactionType.CONTRADICT:
                        contradicts += 1

                if supports >= self.cfg.support_threshold and contradicts == 0:
                    claim.status = ClaimStatus.CONSENSUS
                    surviving.append(claim)
                elif contradicts > 0:
                    claim.status = ClaimStatus.CONTESTED
                    surviving.append(claim)
                else:
                    claim.status = ClaimStatus.ORPHAN          # discarded
        else:
            # No cross-examination: every Round-1 claim moves on.
            surviving = list(trace.all_claims)

        # --- Round 3: graph-mediated arbitration ------------------------------
        if self.cfg.enable_arbiter:
            for claim in surviving:
                claim.grade = self.prior.arbitrate(claim, typed=self.cfg.typed_claims)
                # Peer-consensus claims are trusted; contested ones must pass the
                # arbiter (weak grades are pruned as suspected hallucinations).
                if claim.status == ClaimStatus.CONSENSUS or claim.grade not in self.cfg.drop_grades:
                    claim.status = ClaimStatus.VERIFIED
                    trace.verified_claims.append(claim)
                else:
                    trace.pruned_claims.append(claim)   # suspected hallucination
        else:
            for claim in surviving:
                claim.status = ClaimStatus.VERIFIED
                trace.verified_claims.append(claim)

        return trace
