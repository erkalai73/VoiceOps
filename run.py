#!/usr/bin/env python3
"""Main Entrypoint for PERSON 2 — Semantic Vector Search Support Agent.

Usage:
  Run Evaluation:
    python run.py --mode evaluate

  Launch Web Chat Interface (FastAPI):
    python run.py --mode web --port 8000

  Batch Audio Processing (Official Capstone Requirement):
    python run.py --input-dir PATH_TO_AUDIO_FOLDER --output-dir PATH_TO_OUTPUT_FOLDER
"""
import argparse
import os
import sys

from src.knowledge_base import KnowledgeBase
from src.retriever import SemanticVectorRetriever
from src.tools_adapter import ToolsAdapter
from src.conversation_agent import ConversationAgent
from src.voice_adapter import GnaniVoiceAdapter
from evaluation.eval_questions import evaluate_questions
from evaluation.eval_conversations import evaluate_conversations
from src.cli import run_interactive_cli, run_voice_cli


def main():
    parser = argparse.ArgumentParser(description="PERSON 2: Semantic Vector Search Support Agent (VoiceOps 2026)")
    parser.add_argument('--mode', choices=['evaluate', 'web', 'chat', 'voice', 'batch'], default='evaluate',
                        help="Execution mode (default: evaluate)")
    parser.add_argument('--input-dir', dest='input_dir', type=str, default=None,
                        help="Path to folder of audio files for batch processing")
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='sample_outputs',
                        help="Path to directory where outputs and reports are saved")
    parser.add_argument('--dataset-dir', dest='dataset_dir', type=str, default='Capstone_Project_Dataset_FTTT_2026',
                        help="Path to dataset root folder")
    parser.add_argument('--voice', action='store_true', default=False, help="Speak aloud")
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Host for web server")
    parser.add_argument('--port', type=int, default=8000, help="Port for web server")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, args.dataset_dir) if not os.path.isabs(args.dataset_dir) else args.dataset_dir
    output_path = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_path, exist_ok=True)

    kb = KnowledgeBase(dataset_path)
    retriever = SemanticVectorRetriever(kb, index_path=os.path.join(output_path, 'vector_index.pkl'))
    tools = ToolsAdapter(dataset_path)
    agent = ConversationAgent(kb, tools, retriever)
    voice = GnaniVoiceAdapter(agent)

    if args.input_dir:
        input_audio_path = os.path.join(base_dir, args.input_dir) if not os.path.isabs(args.input_dir) else args.input_dir
        print(f"Running Batch Audio Processing with Gnani.ai...")
        print(f"  Input:  {input_audio_path}")
        print(f"  Output: {output_path}")
        if not voice.is_live:
            print("\n[BLOCKED] Gnani credentials not configured: GNANI_API_KEY environment variable is missing.")
            print("  Please configure GNANI_API_KEY in your .env file or environment.")
            summary = voice.process_batch(input_audio_path, output_path)
            sys.exit(1)
        summary = voice.process_batch(input_audio_path, output_path)
        print(f"Batch Processing Completed:")
        print(f"  Total: {summary['total_files']}, Succeeded: {summary['successful']}, Failed: {summary['failed']}")
        sys.exit(0 if summary['failed'] == 0 else 1)

    if args.mode == 'voice':
        run_voice_cli(agent, voice_adapter=voice)
    elif args.mode == 'chat':
        run_interactive_cli(agent, voice_adapter=voice if args.voice else None)
    elif args.mode == 'web':
        import uvicorn
        from src.web_app import app
        print(f"Starting ZentroMart Support Web Interface at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.mode == 'evaluate':
        print("=" * 60)
        print("  Evaluating Step 1 Baseline vs Step 2 Semantic Vector RAG...")
        print("=" * 60)
        q_summary = evaluate_questions(dataset_path, output_path)
        print("Practice Questions Summary:")
        print(f"  Step 1 Baseline Decision Accuracy: {q_summary['step1_baseline']['decision_accuracy'] * 100:.1f}%")
        print(f"  Step 2 Semantic RAG Decision Acc:  {q_summary['step2_rag']['decision_accuracy'] * 100:.1f}%")
        print(f"  Step 2 Fact Coverage Rate:         {q_summary['step2_rag']['fact_coverage_rate'] * 100:.1f}%")
        print(f"  Step 2 Retrieval Evidence Hit:     {q_summary['step2_rag']['retrieval_evidence_accuracy'] * 100:.1f}%")

        print("\n" + "=" * 60)
        print("  Evaluating Step 3 Multi-Turn Conversations & Tools...")
        print("=" * 60)
        c_summary = evaluate_conversations(dataset_path, output_path)
        print(f"  Conversations:      {c_summary['total_conversations']} ({c_summary['total_turns']} turns)")
        print(f"  Decision Accuracy:  {c_summary['decision_accuracy'] * 100:.1f}%")
        print(f"  Tool-Call Accuracy: {c_summary['tool_call_accuracy'] * 100:.1f}%")
        print(f"  Fact Coverage Rate: {c_summary['fact_coverage_rate'] * 100:.1f}%")
        print("\nAll evaluation artifacts saved to:", output_path)


if __name__ == '__main__':
    main()
