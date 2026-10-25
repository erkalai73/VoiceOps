"""Evaluation pipeline for Step 1 Baseline vs Step 2 Semantic RAG in PERSON 2."""
import csv
import json
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.knowledge_base import KnowledgeBase
from src.retriever import SemanticVectorRetriever
from src.baseline import BaselineAgent
from src.rag_agent import RAGAgent


def evaluate_questions(dataset_dir: str, output_dir: str) -> Dict[str, Any]:
    kb = KnowledgeBase(dataset_dir)
    retriever = SemanticVectorRetriever(kb)
    baseline_agent = BaselineAgent()
    rag_agent = RAGAgent(kb, retriever)

    q_path = os.path.join(dataset_dir, 'practice', 'questions.csv')
    os.makedirs(output_dir, exist_ok=True)

    with open(q_path, encoding='utf-8') as f:
        questions = list(csv.DictReader(f))

    baseline_rows = []
    rag_rows = []

    b_correct_decision = 0
    b_fact_covered = 0
    b_unsupported_abstained = 0

    r_correct_decision = 0
    r_fact_covered = 0
    r_unsupported_abstained = 0
    r_evidence_hits = 0

    unanswerable_total = 0

    for q in questions:
        qid = q['id']
        text = q['question']
        exp_decision = q.get('expected_decision', 'answer').strip()
        acceptable_decisions = [d.strip() for d in q.get('acceptable_decisions', exp_decision).split(';')]
        answerable = q.get('answerable', 'True').strip().lower() == 'true'
        if not answerable:
            unanswerable_total += 1

        must_include_raw = q.get('must_include', '').strip()
        must_include_groups = [g.strip() for g in must_include_raw.split(';') if g.strip()]
        must_not_include_raw = q.get('must_not_include', '').strip()
        must_not_include_items = [item.strip() for item in must_not_include_raw.split(';') if item.strip()]

        gold_evidence_raw = q.get('gold_evidence', '').strip()
        gold_sections = [s.strip() for s in gold_evidence_raw.split(';') if s.strip()]

        # 1. Baseline
        b_res = baseline_agent.answer_question(text)
        b_resp = b_res['response']
        b_dec = b_res.get('decision', 'answer')
        b_facts_ok = True
        if must_include_groups:
            for grp in must_include_groups:
                alts = [alt.strip().lower() for alt in grp.split('/')]
                if not any(alt in b_resp.lower() for alt in alts):
                    b_facts_ok = False
                    break
        else:
            b_facts_ok = False

        b_dec_ok = b_dec in acceptable_decisions
        if b_dec_ok:
            b_correct_decision += 1
        if b_facts_ok:
            b_fact_covered += 1
        if not answerable and any(w in b_resp.lower() for w in ["don't have", "not sure", "colleague"]):
            b_unsupported_abstained += 1

        baseline_rows.append({
            'question_id': qid,
            'question': text,
            'response': b_resp,
            'decision': b_dec,
            'decision_correct': b_dec_ok,
            'facts_covered': b_facts_ok,
            'source': b_res['source']
        })

        # 2. Semantic RAG
        r_res = rag_agent.answer_question(text)
        r_resp = r_res['response']
        r_dec = r_res['decision']
        r_evidence = r_res['evidence_records']

        r_facts_ok = True
        if must_include_groups:
            for grp in must_include_groups:
                alts = [alt.strip().lower() for alt in grp.split('/')]
                if not any(alt in r_resp.lower() for alt in alts):
                    r_facts_ok = False
                    break
        for bad in must_not_include_items:
            if bad.lower() in r_resp.lower():
                r_facts_ok = False
                break

        r_dec_ok = r_dec in acceptable_decisions
        if r_dec_ok:
            r_correct_decision += 1
        if r_facts_ok:
            r_fact_covered += 1

        if not answerable and any(w in r_resp.lower() for w in ["don't have", "not sure", "colleague", "connect"]):
            r_unsupported_abstained += 1

        gold_hit = False
        if gold_sections:
            for g in gold_sections:
                if any(g.lower() in ev.lower() for ev in r_evidence):
                    gold_hit = True
                    break
        else:
            gold_hit = True
        if gold_hit:
            r_evidence_hits += 1

        rag_rows.append({
            'question_id': qid,
            'question': text,
            'response': r_resp,
            'decision': r_dec,
            'decision_correct': r_dec_ok,
            'facts_covered': r_facts_ok,
            'gold_evidence': gold_evidence_raw,
            'retrieved_evidence': " ; ".join(r_evidence),
            'evidence_matched': gold_hit,
            'source': r_res['source']
        })

    with open(os.path.join(output_dir, 'baseline_results.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
        writer.writeheader()
        writer.writerows(baseline_rows)

    with open(os.path.join(output_dir, 'rag_results.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rag_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rag_rows)

    total_q = len(questions)
    summary = {
        'total_questions': total_q,
        'unanswerable_total': unanswerable_total,
        'step1_baseline': {
            'decision_accuracy': round(b_correct_decision / total_q, 3),
            'fact_coverage_rate': round(b_fact_covered / total_q, 3),
            'abstention_rate': round(b_unsupported_abstained / max(unanswerable_total, 1), 3),
        },
        'step2_rag': {
            'decision_accuracy': round(r_correct_decision / total_q, 3),
            'fact_coverage_rate': round(r_fact_covered / total_q, 3),
            'abstention_rate': round(r_unsupported_abstained / max(unanswerable_total, 1), 3),
            'retrieval_evidence_accuracy': round(r_evidence_hits / total_q, 3),
        }
    }

    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    ds_dir = sys.argv[1] if len(sys.argv) > 1 else 'Capstone_Project_Dataset_FTTT_2026'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'sample_outputs'
    res = evaluate_questions(ds_dir, out_dir)
    print(json.dumps(res, indent=2))
