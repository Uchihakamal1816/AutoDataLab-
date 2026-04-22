def _compute_score(email: str, response: str, task_id: str = "task_1") -> float:
    if not response or len(response.strip()) < 10:
        return 0.01

    response_lower = response.lower()

    #  START SAFE (never 0)
    score = 0.01

    # 1. APOLOGY CHECK
    apology_words = ["sorry", "apologize", "apologies", "regret"]
    if any(word in response_lower for word in apology_words):
        score += 0.15

    # 2. SOLUTION CHECK
    solution_words = ["resolve", "fix", "refund", "replace", "escalate",
                      "help", "assist", "process", "arrange", "dispatch",
                      "investigate", "correct", "address"]
    if any(word in response_lower for word in solution_words):
        score += 0.15

    # 3. POLITENESS CHECK
    polite_words = ["thank", "appreciate", "understand", "valued", "pleased"]
    if any(word in response_lower for word in polite_words):
        score += 0.10

    # 4. ISSUE ACKNOWLEDGEMENT
    issue_words = ["delay", "delivery", "order", "issue", "problem",
                   "complaint", "refund", "billing", "inconvenience", "error"]
    matches = sum(1 for word in issue_words if word in response_lower)
    score += min(matches * 0.05, 0.10)

    # 5. LENGTH SCORING
    word_count = len(response.split())
    if word_count < 20:
        score += 0.01
    elif word_count < 50:
        score += 0.05
    elif word_count <= 200:
        score += 0.10
    else:
        score += 0.05

    # 6. STRUCTURE
    greetings = ["dear", "hello", "hi ", "good morning"]
    if any(g in response_lower for g in greetings):
        score += 0.05

    closings = ["sincerely", "regards", "best wishes", "thank you", "warm regards"]
    if any(c in response_lower for c in closings):
        score += 0.05

    if "\n" in response or len(response) > 100:
        score += 0.05

    # 7. RUDE PENALTY
    rude_words = ["not my problem", "not our fault", "impossible",
                  "can't help", "cannot help", "ridiculous", "your fault"]
    if any(word in response_lower for word in rude_words):
        score -= 0.20

    # 8. PROFESSIONAL BONUS
    professional_words = ["please", "certainly", "absolutely", "immediately",
                          "priority", "dedicated", "committed", "ensure"]
    prof_matches = sum(1 for word in professional_words if word in response_lower)
    score += min(prof_matches * 0.03, 0.09)

    # 9. TASK-SPECIFIC
    if task_id == "task_1":
        if any(w in response_lower for w in ["refund", "return", "reimburse"]):
            score += 0.04

    elif task_id == "task_2":
        if any(w in response_lower for w in ["understand your frustration",
                                             "completely understand",
                                             "deeply sorry",
                                             "sincerely apologize"]):
            score += 0.04

    elif task_id == "task_3":
        issues_addressed = 0
        if any(w in response_lower for w in ["wrong item", "incorrect item", "item"]):
            issues_addressed += 1
        if any(w in response_lower for w in ["billing", "charge", "payment"]):
            issues_addressed += 1
        if any(w in response_lower for w in ["support", "response", "team"]):
            issues_addressed += 1
        score += min(issues_addressed * 0.04, 0.09)

    score = round(score, 2)

    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    score = max(0.01, min(0.99, score))

    return score


def grade_response(email: str, response: str, task_id: str = "task_1") -> float:
    return max(0.01, min(0.99, _compute_score(email, response, task_id)))


def grade_easy(email: str, response: str) -> float:
    return max(0.01, min(0.99, _compute_score(email, response, "task_1")))


def grade_medium(email: str, response: str) -> float:
    return max(0.01, min(0.99, _compute_score(email, response, "task_2")))


def grade_hard(email: str, response: str) -> float:
    return max(0.01, min(0.99, _compute_score(email, response, "task_3")))