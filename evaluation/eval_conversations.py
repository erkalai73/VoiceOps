"""Evaluation pipeline for Step 3 Conversations in PERSON 2."""
import csv
import json
import os
import sys
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.knowledge_base import KnowledgeBase
from src.retriever import SemanticVectorRetriever
from src.tools_adapter import ToolsAdapter
from src.conversation_agent import ConversationAgent


def evaluate_conversations(dataset_dir: str, output_dir: str) -> Dict[str, Any]:
    kb = KnowledgeBase(dataset_dir)
    retriever = SemanticVectorRetriever(kb)
    tools = ToolsAdapter(dataset_dir)
    agent = ConversationAgent(kb, tools, retriever)

    c_path = os.path.join(dataset_dir, 'practice', 'conversations.csv')
    os.makedirs(output_dir, exist_ok=True)

    with open(c_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    conversations: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cid = r['conversation_id']
        conversations.setdefault(cid, []).append(r)

    total_turns = len(rows)
    correct_decisions = 0
    correct_tool_calls = 0
    fact_covered = 0
    detailed_results = []

    for cid, turns in conversations.items():
        agent.reset_session(cid)
        conv_turns_res = []

        for t in turns:
            turn_no = int(t['turn'])
            msg = t['customer_message']
            exp_decision = t.get('expected_decision', 'answer').strip()
            exp_tools_raw = t.get('expected_tool_calls', '').strip()
            exp_tools = [tool.strip() for tool in exp_tools_raw.split(';') if tool.strip()]
            must_inc_raw = t.get('must_include', '').strip()
            must_inc_groups = [g.strip() for g in must_inc_raw.split(';') if g.strip()]
            must_not_raw = t.get('must_not_include', '').strip()
            must_not_items = [item.strip() for item in must_not_raw.split(';') if item.strip()]

            res = agent.process_turn(cid, msg, explicit_order_id=t.get('order_id'))
            actual_decision = res['decision']
            actual_tools = res['tool_calls']
            resp_text = res['response']

            dec_ok = actual_decision == exp_decision
            if dec_ok:
                correct_decisions += 1

            tools_ok = True
            if exp_tools:
                for et in exp_tools:
                    tool_base = et.split('(')[0]
                    if not any(tool_base in at for at in actual_tools):
                        tools_ok = False
                        break
            if tools_ok:
                correct_tool_calls += 1

            facts_ok = True
            if must_inc_groups:
                for grp in must_inc_groups:
                    alts = [alt.strip().lower() for alt in grp.split('/')]
                    if not any(alt in resp_text.lower() for alt in alts):
                        facts_ok = False
                        break
            if must_not_items:
                for bad in must_not_items:
                    if bad.lower() in resp_text.lower():
                        facts_ok = False
                        break
            if facts_ok:
                fact_covered += 1

            conv_turns_res.append({
                'turn': turn_no,
                'customer_message': msg,
                'assistant_response': resp_text,
                'expected_decision': exp_decision,
                'actual_decision': actual_decision,
                'decision_correct': dec_ok,
                'expected_tools': exp_tools,
                'actual_tools': actual_tools,
                'tools_correct': tools_ok,
                'facts_covered': facts_ok,
                'verified': res['verified']
            })

        detailed_results.append({
            'conversation_id': cid,
            'scenario': turns[0].get('scenario', ''),
            'turns': conv_turns_res
        })

    conv_summary = {
        'total_conversations': len(conversations),
        'total_turns': total_turns,
        'decision_accuracy': round(correct_decisions / total_turns, 3),
        'tool_call_accuracy': round(correct_tool_calls / total_turns, 3),
        'fact_coverage_rate': round(fact_covered / total_turns, 3),
    }

    out_json = os.path.join(output_dir, 'conversation_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'summary': conv_summary, 'conversations': detailed_results}, f, indent=2, ensure_ascii=False)

    return conv_summary


if __name__ == '__main__':
    ds_dir = sys.argv[1] if len(sys.argv) > 1 else 'Capstone_Project_Dataset_FTTT_2026'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'sample_outputs'
    res = evaluate_conversations(ds_dir, out_dir)
    print(json.dumps(res, indent=2))
