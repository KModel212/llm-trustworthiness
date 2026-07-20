import re
from typing import Tuple


class PromptInjectionDetector:
    """
    Detects attempts to hijack the model's system instructions.
    Transform tasks such as translation and quotation are allowed to contain
    unsafe-looking text without being treated as active instructions.
    """

    def __init__(self):
        self.quoted_transform_pattern = re.compile(
            r"^\s*(translate|\u0e41\u0e1b\u0e25\S*|summari[sz]e|quote|classify|label|"
            r"what does\b|explain (?:the )?(?:phrase|sentence|text))\b"
            r".{0,80}[:\"']",
            re.IGNORECASE | re.DOTALL,
        )
        self.injection_patterns = [
            re.compile(
                r"(?:\u0e41\u0e2a\u0e14\u0e07|\u0e40\u0e1b\u0e34\u0e14\u0e40\u0e1c\u0e22|"
                r"\u0e1a\u0e2d\u0e01|\u0e17\u0e33\u0e0b\u0e49\u0e33).{0,40}"
                r"(?:\u0e04\u0e33\u0e2a\u0e31\u0e48\u0e07\u0e23\u0e30\u0e1a\u0e1a|"
                r"\u0e1e\u0e23\u0e2d\u0e21\u0e1b\u0e4c\u0e23\u0e30\u0e1a\u0e1a|"
                r"\u0e1e\u0e23\u0e2d\u0e21\u0e1e\u0e4c\u0e23\u0e30\u0e1a\u0e1a|"
                r"\u0e19\u0e42\u0e22\u0e1a\u0e32\u0e22.{0,15}\u0e0b\u0e48\u0e2d\u0e19|"
                r"\u0e02\u0e49\u0e2d\u0e04\u0e27\u0e32\u0e21.{0,15}\u0e0b\u0e48\u0e2d\u0e19)",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\b(print|show|reveal|display|leak|repeat|บอก|เปิดเผย|แสดง)\b.{0,40}"
                r"\b(system prompt|hidden (?:prompt|policy|instructions?)|developer message|"
                r"system instructions?|คำสั่งระบบ|พรอมป์ระบบ|นโยบายที่ซ่อนอยู่)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\b(ignore|disregard|forget|override)\b.{0,20}\b(previous|all|your|above)\b.{0,20}"
                r"\b(instructions?|directions?|prompts?|guidelines?|rules?)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\b(ignore|disregard|forget|override)\b.{0,20}\b(previous|all)\b.{0,20}"
                r"\b(instructions?)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\bbase64\s+request\b.{0,80}\b[a-zA-Z0-9+/]{24,}={0,2}\b",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"(ลืม|ไม่ต้องสนใจ|เพิกเฉย|ข้าม|ยกเลิก).{0,30}"
                r"(คำสั่ง|กฎ|ข้อจำกัด|พรอมป์).{0,30}(ก่อนหน้า|ทั้งหมด|ระบบ)",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\b(you are now|act as|become|pretend (you are|to be))\b.{0,30}"
                r"\b(unrestricted|unfiltered|without (restrictions?|limits?|rules?)|"
                r"no restrictions|jailbroken|god mode|admin mode|root)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"(สมมติว่า|จินตนาการว่า).{0,30}(ไม่มีข้อจำกัด|ไร้ข้อจำกัด|ไม่มีกฎ)",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(r"<\|im_start\|>.*?<\|im_end\|>", re.IGNORECASE | re.DOTALL),
            re.compile(r"\[/?system\]", re.IGNORECASE),
            re.compile(r"<\|system\|>|<<SYS>>|\[INST\]", re.IGNORECASE),
            re.compile(
                r"\b(new instruction[s]?|updated (system )?prompt|revised instructions?)"
                r"\s*[:\-]\s*",
                re.IGNORECASE,
            ),
        ]

    def detect(self, normalized_text: str) -> Tuple[bool, float]:
        """Returns (is_injected, confidence_score [0.0-1.0])."""
        if self.quoted_transform_pattern.search(normalized_text):
            return False, 0.0

        matches = sum(
            1 for pattern in self.injection_patterns
            if pattern.search(normalized_text)
        )

        if matches > 0:
            confidence = min(1.0, 0.6 + (matches - 1) * 0.2)
            return True, confidence

        return False, 0.0
