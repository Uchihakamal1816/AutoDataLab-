from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    immediate: float
    cumulative: float
    terminal_grader: Optional[float] = None


class CoSAction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action_type: Literal["consult", "ask", "summarize", "submit", "noop"]
    expert_id: Optional[Literal["analyst", "finance", "hr", "strategy"]] = None
    sub_question_id: Optional[str] = None
    notes: Optional[str] = None


class ExpertReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expert_id: Literal["analyst", "finance", "hr", "strategy"]
    title: str
    summary: str
    metrics: Dict[str, float | int | str] = Field(default_factory=dict)
    bullet_points: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    memo: Optional[str] = None
    score: Optional[float] = None


class Brief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    metrics: Dict[str, float | int | str] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    hr_memo: str = ""
    consulted_experts: List[str] = Field(default_factory=list)


class CoSObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    reward: float = 0.0
    instruction: str
    history: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    data_quality_score: float = 0.0
    task_name: str
    task_difficulty: str
    max_steps: int = 12
    step_count: int = 0
    available_experts: List[str] = Field(default_factory=lambda: ["analyst", "finance", "hr", "strategy"])
    consulted_experts: List[str] = Field(default_factory=list)
    expert_reports: Dict[str, ExpertReport] = Field(default_factory=dict)
    current_brief: Optional[Brief] = None
    reward_breakdown: Optional[RewardBreakdown] = None
    terminal_grader_score: Optional[float] = None


class CoSState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_name: str
    step_count: int = 0
    done: bool = False
    consulted_experts: List[str] = Field(default_factory=list)
    expert_reports: Dict[str, ExpertReport] = Field(default_factory=dict)
    current_brief: Optional[Brief] = None
    cumulative_reward: float = 0.0
