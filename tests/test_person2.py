"""Automated Test Suite for PERSON 2 (Semantic Vector Search Agent)."""
import os
import unittest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge_base import KnowledgeBase
from src.retriever import SemanticVectorRetriever
from src.tools_adapter import ToolsAdapter
from src.conversation_agent import ConversationAgent
from src.voice_adapter import GnaniVoiceAdapter


class TestPerson2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.ds_dir = os.path.join(base_dir, 'Capstone_Project_Dataset_FTTT_2026')
        cls.kb = KnowledgeBase(cls.ds_dir)
        cls.index_file = os.path.join(base_dir, 'sample_outputs', 'test_vector_index.pkl')
        cls.retriever = SemanticVectorRetriever(cls.kb, index_path=cls.index_file)
        cls.tools = ToolsAdapter(cls.ds_dir)
        cls.conv_agent = ConversationAgent(cls.kb, cls.tools, cls.retriever)

    def test_semantic_vector_retrieval(self):
        """Test semantic vector matching and index persistence."""
        results = self.retriever.retrieve("What is the return window for clothing and sneakers?", top_k=5)
        self.assertGreater(len(results), 0)
        top_sec, score = results[0]
        # RET-2 contains clothing, footwear and bags
        ret_ids = [s.section_id for s, _ in results]
        self.assertIn("RET-2", ret_ids)
        self.assertGreater(score, 0.0)

    def test_metadata_status_filter(self):
        """Ensure superseded policies are filtered or demoted in normal queries."""
        results = self.retriever.retrieve("electronics return policy", top_k=5)
        ret_ids = [s.section_id for s, _ in results]
        self.assertTrue("RET-3" in ret_ids or "RET-2" in ret_ids)
        # RETv3-2 should not be top 1
        self.assertNotEqual(ret_ids[0], "RETv3-2")

    def test_identity_gated_lookup(self):
        """Test that order lookup gates details behind phone/pincode verification."""
        unverified = self.tools.lookup("18599253")
        self.assertTrue(unverified.get('verification_required'))
        self.assertNotIn('items', unverified)

        verified = self.tools.lookup("18599253", verification_value="4265")
        self.assertTrue(verified.get('verified'))
        self.assertIn('items', verified)

    def test_fastapi_web_chat(self):
        """Test FastAPI endpoints for web chat."""
        from fastapi.testclient import TestClient
        from src.web_app import app
        client = TestClient(app)
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("ZentroMart Support AI", res.text)

        chat_res = client.post("/api/chat", json={"message": "hello", "session_id": "test_s1"})
        self.assertEqual(chat_res.status_code, 200)
        data = chat_res.json()
        self.assertIn("response", data)

    def test_mocked_voice_adapter_pipeline(self):
        """[MOCKED UNIT TEST] Verify voice adapter audio processing workflow with mocked Gnani clients."""
        from unittest.mock import MagicMock
        mock_stt = MagicMock()
        mock_tts = MagicMock()
        mock_stt.transcribe.return_value = {"transcript": "What is the return window for clothing?"}
        mock_tts.synthesize.return_value = b"RIFFfakeaudio"

        voice = GnaniVoiceAdapter(self.conv_agent, stt_client=mock_stt, tts_client=mock_tts)
        self.assertTrue(voice.is_live)

        res = voice.process_voice_turn("/tmp/test_voice_turn.wav", conv_id="test_mock_voice")
        self.assertEqual(res['transcription'], "What is the return window for clothing?")
        self.assertEqual(res['stt_status'], "live_gnani_stt")
        self.assertIn("zentromart", res['reply_text'].lower())
        mock_tts.synthesize.assert_called_once()


if __name__ == '__main__':
    unittest.main()
