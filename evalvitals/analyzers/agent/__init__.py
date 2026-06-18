"""Agent failure analyzers — operate on a FailureCase's Trajectory.

Build order (cheapest/most reliable first): deterministic heuristics
(loop_detect, ignored_obs) → LLM-judge (first_error_judge) → causal replay
(counterfactual, Stage 2).
"""

from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay
from evalvitals.analyzers.agent.first_error_judge import FirstErrorJudge
from evalvitals.analyzers.agent.ignored_obs import IgnoredObservationDetector
from evalvitals.analyzers.agent.loop_detect import LoopDetector

__all__ = ["LoopDetector", "IgnoredObservationDetector", "FirstErrorJudge", "CounterfactualReplay"]
