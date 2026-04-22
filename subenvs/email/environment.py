from typing import Tuple
from subenvs.email.models import EmailObservation, EmailAction, EmailState
from subenvs.email.graders import grade_response

TASKS = {
    "task_1": "Customer is asking for a simple refund for a damaged item.",
    "task_2": "Customer is angry about delayed delivery and threatening to leave a bad review.",
    "task_3": "Customer has multiple complaints: wrong item received, billing error, and no response from support for 2 weeks.",
}

class EmailEnv:
    def __init__(self, task_id: str = "task_1"):
        self._state = None
        self.task_id = task_id

    def reset(self) -> EmailObservation:
        email = TASKS[self.task_id]
        self._state = EmailState(
            current_email=email,
            step_count=0,
            done=False,
            task_id=self.task_id
        )
        return EmailObservation(email=email, step_count=0, task_id=self.task_id)

    def step(self, action: EmailAction) -> Tuple[EmailObservation, float, bool, dict]:

        if self._state is None:
            return (
                EmailObservation(
                    email="",
                    step_count=0,
                    task_id=self.task_id
                ),
                0.01,
                True,
                {"score": 0.01}
            )
        if self._state and self._state.done:
            return (
                EmailObservation(
                    email=self._state.current_email,
                    step_count=self._state.step_count,
                    task_id=self._state.task_id
                ),
                0.01,
                True,
                {"score": 0.01}
            )

        self._state.step_count += 1
        score = grade_response(self._state.current_email, action.response, self.task_id)
        self._state.done = True

        obs = EmailObservation(
            email=self._state.current_email,
            step_count=self._state.step_count,
            task_id=self._state.task_id
        )
        score = max(0.01, min(0.99, score))
        score = round(score, 2)

        if score <= 0.0:
            score = 0.01
        if score >= 1.0:
            score = 0.99

        return obs, score, True, {"score": score}

    def state(self) -> EmailState:
        return self._state