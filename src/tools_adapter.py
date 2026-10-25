"""Step 3 Tools Adapter for PERSON 2."""
import sys
import os
import re
from typing import Dict, Any, Optional


class ToolsAdapter:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = os.path.abspath(dataset_dir)
        tools_path = os.path.join(self.dataset_dir, 'tools')
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)

        try:
            from tools import lookup_order, create_ticket
            self._lookup_order_fn = lookup_order
            self._create_ticket_fn = create_ticket
        except ImportError:
            self._lookup_order_fn = None
            self._create_ticket_fn = None

    def lookup(self, order_id: str, verification_value: Optional[str] = None) -> Dict[str, Any]:
        if not self._lookup_order_fn:
            return {'found': False, 'message': 'Tools module not found'}
        clean_id = re.sub(r'\D', '', str(order_id))
        clean_proof = re.sub(r'\D', '', str(verification_value)) if verification_value else None
        return self._lookup_order_fn(clean_id, verification_value=clean_proof)

    def ticket(self, summary: str, order_id: Optional[str] = None, reason: Optional[str] = None, intent: Optional[str] = None) -> Dict[str, Any]:
        if not self._create_ticket_fn:
            return {'created': False, 'error': 'Tools module not found'}
        safe_summary = summary.strip()
        if len(safe_summary) < 20:
            safe_summary = f"Escalation handover: {safe_summary}. Human assistance required."
        safe_summary = re.sub(r'\b\d{6}\b', '[REDACTED_PIN]', safe_summary)
        safe_summary = re.sub(r'\b\d{10}\b', '[REDACTED_PHONE]', safe_summary)
        return self._create_ticket_fn(summary=safe_summary, order_id=order_id, reason=reason, intent=intent)
