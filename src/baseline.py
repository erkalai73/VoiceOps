"""Step 1 Baseline Agent for PERSON 2."""
from typing import Dict, Any
from .llm_adapter import LLMAdapter

BASELINE_SYSTEM_PROMPT = """You are an AI customer support assistant for ZentroMart, an Indian online shopping store.
Always identify yourself as an AI assistant.
Answer customer questions to the best of your general knowledge without internal company documents.
"""


class BaselineAgent:
    def __init__(self, llm: LLMAdapter = None):
        self.llm = llm or LLMAdapter()
    def answer_question(self, question: str) -> Dict[str, Any]:
        prompt = f"Customer Question: {question}\n\nPlease provide your answer as ZentroMart's AI Support Assistant:"
        resp_raw = self.llm.generate(prompt=prompt, system_prompt=BASELINE_SYSTEM_PROMPT, context=None)
        response_text = resp_raw[0] if isinstance(resp_raw, tuple) else resp_raw
        decision = "answer"
        return {
            "question": question,
            "response": response_text,
            "decision": decision,
            "is_rag": False,
            "retrieved_evidence": [],
            "retrieved_sections": [],
            "source": resp_raw[1] if isinstance(resp_raw, tuple) and len(resp_raw) > 1 else "offline_baseline",
            "has_retrieval": False,
            "grounded": False
        }
