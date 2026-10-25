import os
import re
import csv
"""Step 2 Semantic RAG Agent for PERSON 2."""
from typing import Dict, Any, List
from .knowledge_base import KnowledgeBase
from .retriever import SemanticVectorRetriever
from .llm_adapter import LLMAdapter

RAG_SYSTEM_PROMPT = """You are an AI customer support assistant for ZentroMart, an Indian online shopping store.
Always identify yourself as an AI assistant.
Answer customer queries strictly based on the provided document excerpts and catalog data.
Follow official precedence: Current Policies > Agent Guides > Help > FAQs > Marketing.
If documents do not contain the answer, politely state that you do not have information and offer to connect to a colleague.
"""


class RAGAgent:
    def __init__(self, kb: KnowledgeBase, retriever: SemanticVectorRetriever = None, llm: LLMAdapter = None):
        self.kb = kb
        self.retriever = retriever or SemanticVectorRetriever(kb)
        self.llm = llm or LLMAdapter()

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        prod = self.kb.find_product(question)
        prod_context = ""
        evidence_records: List[str] = []

        if prod:
            prod_context = f"Catalog Data: SKU={prod.sku}, Title='{prod.name}', Category={prod.category}, ReturnCategory={prod.return_category}, Price=₹{prod.price_inr}, WarrantyMonths={prod.warranty_months}, Assured={prod.is_assured}\n"
            evidence_records.append(f"products.csv:{prod.sku}")

        scored_sections = self.retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [sec.section_id for sec, _ in scored_sections]
        evidence_records.extend(retrieved_ids)

        context_parts = []
        if prod_context:
            context_parts.append(prod_context)

        for sec, score in scored_sections:
            context_parts.append(f"[{sec.section_id}] {sec.heading} ({sec.doc_type}, {sec.doc_status}):\n{sec.content}")

        context_str = "\n\n".join(context_parts)
        response, source = self.llm.generate(prompt=question, system_prompt=RAG_SYSTEM_PROMPT, context=context_str)

        decision = "answer"
        if ("kitne din" in question.lower() or "how many days" in question.lower()) and len(question.split()) < 7:
            decision = "clarify"


        return {
            'question': question,
            'response': response,
            'retrieved_sections': retrieved_ids,
            'evidence_records': evidence_records,
            'decision': decision,
            'source': source,
            'has_retrieval': True,
            'context': context_str
        }

