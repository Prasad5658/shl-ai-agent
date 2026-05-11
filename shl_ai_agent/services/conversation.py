from services.llm import generate_reply
from services.guardrails import clean_user_text, guardrail_reply, is_prompt_injection
from services.retriever import find_assessments_by_name, retrieve_assessments


ROLE_HINTS = {
    "analyst",
    "consultant",
    "developer",
    "engineer",
    "graduate",
    "java",
    "manager",
    "sales",
    "supervisor",
    "support",
}

SKILL_HINTS = {
    "aptitude",
    "cognitive",
    "communication",
    "coding",
    "java",
    "leadership",
    "management",
    "personality",
    "reasoning",
    "stakeholder",
}

COMPARISON_MARKERS = ("compare", "comparison", "difference", "different", "between", " vs ", " versus ")
REFINEMENT_MARKERS = ("actually", "also", "add", "include", "instead", "remove", "narrow", "only", "prefer")


def _user_messages(messages):
    return [clean_user_text(message["content"]) for message in messages if message.get("role") == "user"]


def _has_enough_signal(text):
    lowered = text.lower()
    has_role = any(hint in lowered for hint in ROLE_HINTS)
    has_skill = any(hint in lowered for hint in SKILL_HINTS)
    return has_role or (has_skill and len(lowered.split()) >= 4)


def _is_comparison(text):
    lowered = f" {text.lower()} "
    return any(marker in lowered for marker in COMPARISON_MARKERS)


def _is_refinement(text, messages):
    if len(_user_messages(messages)) < 2:
        return False

    lowered = text.lower()
    return any(marker in lowered for marker in REFINEMENT_MARKERS)


def _format_recommendations(results):
    recommendations = []

    for item in results:
        recommendations.append(
            {
                "name": item["name"],
                "url": item["url"],
                "test_type": item["test_type"],
                "duration": item["duration"],
                "remote": item["remote"],
                "adaptive": item["adaptive"],
                "score": item.get("_score"),
            }
        )

    return recommendations


def _context_from_results(results):
    context = ""

    for item in results:
        context += f"""
        Name: {item['name']}
        Description: {item['description']}
        Test Type: {item['test_type']}
        Duration: {item['duration']}
        Remote: {item['remote']}
        Adaptive: {item['adaptive']}
        """

    return context


def _fallback_recommendation_reply(results, refined=False):
    if not results:
        return "I could not find a strong catalog match. Please share the role, seniority, and key skills."

    names = ", ".join(item["name"] for item in results[:3])
    prefix = "I updated the shortlist" if refined else "I found a grounded shortlist"
    return f"{prefix}: {names}. The structured results include SHL links, categories, duration, remote support, and adaptive status."


def _comparison_terms(text):
    lowered = text.lower()

    if " between " in lowered and " and " in lowered:
        after_between = lowered.split(" between ", 1)[1]
        left, right = after_between.split(" and ", 1)
        return [left.strip(" ?."), right.strip(" ?.")]

    for separator in (" vs ", " versus "):
        if separator in lowered:
            parts = [part.strip(" ?.") for part in lowered.split(separator) if part.strip(" ?.")]
            if len(parts) >= 2:
                return parts[:2]

    return [text]


def _handle_comparison(latest_message):
    terms = _comparison_terms(latest_message)
    results = []

    for term in terms:
        results.extend(find_assessments_by_name(term, top_k=2))

    seen = set()
    unique_results = []

    for item in results:
        key = item["name"]
        if key not in seen:
            unique_results.append(item)
            seen.add(key)

    context = _context_from_results(unique_results[:4])

    prompt = f"""
    Compare the SHL assessments using only this retrieved catalog context.
    If the catalog context is insufficient, say what is known and what is not known.

    User question:
    {latest_message}

    Catalog context:
    {context}
    """

    fallback = (
        "Here is a grounded comparison from the catalog results. "
        "Use the names, categories, duration, remote support, and descriptions below; "
        "avoid assuming details that are not present in the catalog."
    )

    return {
        "reply": generate_reply(prompt, fallback=fallback),
        "recommendations": _format_recommendations(unique_results[:4]),
        "end_of_conversation": True,
    }


def process_conversation(messages):
    latest_message = clean_user_text(messages[-1]["content"])

    if is_prompt_injection(latest_message):
        return guardrail_reply()

    if _is_comparison(latest_message):
        return _handle_comparison(latest_message)

    if not _has_enough_signal(latest_message):
        return {
            "reply": (
                "Could you share the role, seniority level, key skills, and whether you need "
                "ability, coding, personality, or behavioral assessments?"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    refined = _is_refinement(latest_message, messages)
    user_history = _user_messages(messages)
    retrieval_query = " ".join(user_history) if refined else latest_message

    results = retrieve_assessments(retrieval_query, top_k=10)
    recommendation_list = _format_recommendations(results)
    context = _context_from_results(results)

    prompt = f"""
    You are an SHL assessment recommender.
    Use only the retrieved SHL catalog context.
    Do not invent assessment features that are absent from the context.
    If this is a refinement, update the shortlist using the full conversation.

    User query:
    {retrieval_query}

    Retrieved SHL assessments:
    {context}

    Generate a concise recommendation reply. Mention why the top options fit.
    """

    reply = generate_reply(prompt, fallback=_fallback_recommendation_reply(results, refined=refined))

    return {
        "reply": reply,
        "recommendations": recommendation_list,
        "end_of_conversation": True,
    }
