# 2-Minute Live Demo Instructions — PERSON 2
## VoiceOps 2026: Semantic Vector Search Support Agent

Demonstrates semantic vector search, browser web app, multi-turn state, and Gnani voice capabilities.

---

## Demo Script (Timeline: 120 Seconds)

### Scene 1: Web Interface & Semantic Retrieval (0:00 – 0:35)
1. Start the server:
   ```bash
   python run.py --mode web --port 8000
   ```
2. Open `http://localhost:8000` in browser.
3. Enter a query with non-standard phrasing:
   > *"If I don't want to keep my headphones, how many days do I have to send them back?"*
4. Show that the Semantic Vector Retriever matches `[RET-2]` (return windows for electronics) despite using "send them back" instead of "reverse pickup".
5. Note the displayed evidence tags at the bottom of the assistant message.

---

### Scene 2: Multi-Turn Identity Verification (0:35 – 1:05)
1. In the chat, type:
   > *"Hello, mujhe order 1859 9253 ke baare mein poochna hai."*
2. Assistant replies:
   > *"Thank you for sharing order 18599253. To verify your identity, please provide the last 4 digits of your registered phone number or your 6-digit delivery pincode."*
3. Provide verification proof:
   > *"Last four digits 4265."*
4. Assistant verifies the customer and unlocks private order state without repeating the proof.

---

### Scene 3: Refund Timeline Calculation & Precedence (1:05 – 1:35)
1. Customer asks:
   > *"Aur refund kitne din mein aayega?"*
2. Assistant responds:
   > *"Wallet payments ka refund instant hota hai, initiate hone ke 2 ghante (2 hours) ke andar ZentroMart Wallet mein credit ho jaata hai."*
3. Highlight that the agent correctly applied `[REF-2]` and `[PLS-2]` over `[FAQ-1]`.

---

### Scene 4: Voice Pipeline via Gnani.ai (1:35 – 2:00)
1. Run the official batch audio runner:
   ```bash
   python run.py --input-dir sample_outputs/test_audio --output-dir sample_outputs/batch_audio_results
   ```
2. Inspect `sample_outputs/batch_audio_results/batch_processing_summary.json` showing Gnani STT transcriptions and TTS output WAV generation.
