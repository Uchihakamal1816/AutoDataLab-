from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import pandas as pd

from .experts import DataAnalystExpert, FinanceExpert, HRExpert, StrategyExpert
from .graders import grade_episode, load_metric_ground_truth
from .models import Brief, CoSAction, CoSObservation, CoSState, ExpertReport, RewardBreakdown

TASK_ROOT = Path(__file__).resolve().parent / 'tasks'

# Per-task order for “single” baselines, oracle, and any policy that should cover required experts.
REQUIRED_EXPERTS_BY_TASK: dict[str, list[str]] = {
    'easy_brief': ['analyst', 'finance', 'hr'],
    'medium_brief': ['analyst', 'finance', 'strategy', 'hr'],
    'hard_brief': ['analyst', 'finance', 'strategy', 'hr'],
    'expert_brief': ['analyst', 'finance', 'strategy', 'hr'],
    'risk_brief': ['analyst', 'finance', 'strategy', 'hr'],
    'crisis_brief': ['analyst', 'finance', 'strategy', 'hr'],
}


def required_experts_for_task(task_name: str) -> list[str]:
    return list(REQUIRED_EXPERTS_BY_TASK.get(task_name, ['analyst', 'finance', 'hr']))


class CEOBriefEnvironment:
    def __init__(self) -> None:
        self.analyst = DataAnalystExpert()
        self.finance = FinanceExpert()
        self.hr = HRExpert()
        self.strategy = StrategyExpert()
        self.use_rag = False
        self.reset()

    def reset(self, task: str = 'easy_brief', episode_id: str | None = None, use_rag: bool = False) -> CoSObservation:
        self.use_rag = use_rag
        self.episode_id = episode_id or str(uuid4())
        self.task_name = task if (TASK_ROOT / task).exists() else 'easy_brief'
        task_dir = TASK_ROOT / self.task_name
        self.raw_df = pd.read_csv(task_dir / 'raw.csv')
        self.gt_metrics = load_metric_ground_truth(str(task_dir / 'ground_truth.csv')) if (task_dir / 'ground_truth.csv').exists() else {}
        with open(task_dir / 'metadata.json', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.expert_reports: Dict[str, ExpertReport] = {}
        self.current_brief: Brief | None = None
        self.history: list[str] = []
        self.last_reward = 0.0
        self.last_terminal = None
        self.last_data_quality = 0.0
        self.last_issues = ['No experts consulted yet.']
        return self._observe(initial=True)

    def state(self) -> CoSState:
        return CoSState(
            episode_id=self.episode_id,
            task_name=self.task_name,
            step_count=self.step_count,
            done=self.done,
            rag_enabled=self.use_rag,
            consulted_experts=list(self.expert_reports.keys()),
            expert_reports=self.expert_reports,
            current_brief=self.current_brief,
            cumulative_reward=self.cumulative_reward,
        )

    def _observe(self, initial: bool = False) -> CoSObservation:
        return CoSObservation(
            done=self.done,
            reward=0.0 if initial else self.last_reward,
            instruction=self.meta['instruction'],
            history=list(self.history),
            issues=list(self.last_issues),
            data_quality_score=self.last_data_quality,
            task_name=self.task_name,
            task_difficulty=self.meta['difficulty'],
            max_steps=int(self.meta.get('max_steps', 12)),
            step_count=self.step_count,
            rag_enabled=self.use_rag,
            consulted_experts=list(self.expert_reports.keys()),
            expert_reports=self.expert_reports,
            current_brief=self.current_brief,
            reward_breakdown=RewardBreakdown(
                immediate=self.last_reward,
                cumulative=self.cumulative_reward,
                terminal_grader=self.last_terminal,
            ),
            terminal_grader_score=self.last_terminal,
        )

    def _compose_brief(self) -> Brief:
        metrics: Dict[str, Any] = {}
        recommendations: list[str] = []
        summary_parts: list[str] = []
        for expert_id in ('analyst', 'finance'):
            report = self.expert_reports.get(expert_id)
            if report:
                metrics.update(report.metrics)
                summary_parts.append(report.summary)
        if 'strategy' in self.expert_reports:
            recommendations = list(self.expert_reports['strategy'].bullet_points)
            summary_parts.append(self.expert_reports['strategy'].summary)
        hr_memo = self.expert_reports['hr'].memo if 'hr' in self.expert_reports and self.expert_reports['hr'].memo else ''
        summary = ' '.join(summary_parts) if summary_parts else 'No brief drafted yet.'
        self.current_brief = Brief(
            summary=summary,
            metrics=metrics,
            recommendations=recommendations,
            hr_memo=hr_memo,
            consulted_experts=list(self.expert_reports.keys()),
        )
        return self.current_brief

    def _run_expert(self, expert_id: str, focused: bool = False) -> ExpertReport:
        question = self.meta['instruction']
        if expert_id == 'analyst':
            report = self.analyst.run(
                self.task_name, question, self.raw_df, focused=focused, use_rag=self.use_rag
            )
            self.last_data_quality = float(report.metrics.get('data_quality_score', 0.0))
            self.last_issues = report.issues or ['analyst:no material issues']
            return report
        if expert_id == 'finance':
            analyst = self.expert_reports.get('analyst') or self._run_expert('analyst')
            return self.finance.run(
                self.task_name,
                question,
                self.raw_df,
                analyst.metrics,
                self.meta,
                focused=focused,
                use_rag=self.use_rag,
            )
        if expert_id == 'strategy':
            analyst = self.expert_reports.get('analyst') or self._run_expert('analyst')
            finance = self.expert_reports.get('finance') or self._run_expert('finance')
            return self.strategy.run(
                self.task_name, self.meta, analyst, finance, focused=focused, use_rag=self.use_rag
            )
        if expert_id == 'hr':
            analyst = self.expert_reports.get('analyst') or self._run_expert('analyst')
            finance = self.expert_reports.get('finance') or self._run_expert('finance')
            strategy = self.expert_reports.get('strategy')
            return self.hr.run(
                self.task_name, self.meta, analyst, finance, strategy, focused=focused, use_rag=self.use_rag
            )
        raise ValueError(f'Unknown expert {expert_id!r}')

    def _ensure_required_experts(self) -> list[str]:
        """Run any task-required experts that the policy never consulted.

        This guarantees the strategist (and any other required role) always
        contributes to the brief, so the UI / grader always has their report.
        Returns the list of expert ids that were auto-filled.
        """
        auto: list[str] = []
        for expert_id in required_experts_for_task(self.task_name):
            if expert_id in self.expert_reports:
                continue
            try:
                self.expert_reports[expert_id] = self._run_expert(expert_id)
                auto.append(expert_id)
            except Exception:
                continue
        return auto

    def step(self, action: CoSAction) -> CoSObservation:
        if self.done:
            return self._observe()
        self.step_count += 1
        immediate = -0.02
        details = action.model_dump(exclude_none=True)
        self.history.append(json.dumps(details, sort_keys=True))
        if action.action_type in {'consult', 'ask'}:
            if not action.expert_id:
                immediate -= 0.03
                self.last_issues = ['action_missing_expert']
            else:
                prior = action.expert_id in self.expert_reports
                report = self._run_expert(action.expert_id, focused=action.action_type == 'ask')
                self.expert_reports[action.expert_id] = report
                immediate += 0.10 if not prior and action.expert_id in self.meta.get('required_experts', []) else 0.02
                if prior:
                    immediate -= 0.05
                self.last_issues = report.issues or [f'{action.expert_id}:ok']
        elif action.action_type == 'summarize':
            self._ensure_required_experts()
            self._compose_brief()
            immediate += 0.04 if len(self.expert_reports) >= 2 else -0.02
            self.last_issues = ['brief_composed']
        elif action.action_type == 'submit':
            auto_filled = self._ensure_required_experts()
            if self.current_brief is None or auto_filled:
                self._compose_brief()
            self.done = True
            self.last_terminal = grade_episode(
                self.gt_metrics, self.meta, self.current_brief, self.expert_reports, use_rag=self.use_rag
            )
            immediate += self.last_terminal
            self.last_issues = ['submitted'] + (
                [f'auto_consulted:{",".join(auto_filled)}'] if auto_filled else []
            )
        else:
            self.last_issues = ['noop']
            immediate -= 0.01

        if not self.done and self.step_count >= int(self.meta.get('max_steps', 12)):
            auto_filled = self._ensure_required_experts()
            if self.current_brief is None or auto_filled:
                self._compose_brief()
            self.done = True
            self.last_terminal = grade_episode(
                self.gt_metrics, self.meta, self.current_brief, self.expert_reports, use_rag=self.use_rag
            )
            immediate += self.last_terminal
            self.last_issues = ['forced_termination:max_steps'] + (
                [f'auto_consulted:{",".join(auto_filled)}'] if auto_filled else []
            )

        self.last_reward = round(immediate, 4)
        self.cumulative_reward = round(self.cumulative_reward + self.last_reward, 4)
        return self._observe()


def oracle_action_for_observation(obs: CoSObservation) -> CoSAction:
    for expert in required_experts_for_task(obs.task_name):
        if expert not in obs.consulted_experts:
            return CoSAction(action_type='consult', expert_id=expert)
    if obs.current_brief is None:
        return CoSAction(action_type='summarize')
    return CoSAction(action_type='submit')
