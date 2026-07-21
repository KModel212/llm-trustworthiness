import re


class SafetyContext:
    """
    Detects contexts where harmful terms are likely being discussed safely.

    This is deliberately small and intent-focused. It reduces false refusals for
    education, translation, summarization, analysis, history, prevention, and
    defensive cybersecurity without treating fictional/actionable requests as
    safe.
    """

    def __init__(self):
        transform = (
            r"\b(translate|แปล|summari[sz]e|สรุป|classify|จัดประเภท|label|"
            r"quote|ยกคำพูด|what does|หมายความว่า|อธิบายคำว่า)\b"
        )
        educational = (
            r"\b(explain|อธิบาย|overview|ภาพรวม|history|ประวัติ|historical|"
            r"หลักการ|เชิงฟิสิกส์|เชิงการแพทย์|เชิงกฎหมาย|เชิงประวัติศาสตร์|"
            r"research|ศึกษา|เพื่อการศึกษา|วิเคราะห์|analysis|analyze|วิเคราะห์|"
            r"ทำงานอย่างไร|ระดับสูง|จัดประเภท)\b"
        )
        defensive = (
            r"\b(prevent|ป้องกัน|detect|ตรวจจับ|defend|defense|defence|"
            r"recognize|สังเกต|report|แจ้ง|mitigate|ลดความเสี่ยง|"
            r"security awareness|ความปลอดภัยไซเบอร์|ควรช่วย|ช่วยอย่างไร|"
            r"โดยไม่ให้ขั้นตอนโจมตี)\b"
        )
        self.safe_context_pattern = re.compile(
            rf"(?:{transform}|{educational}|{defensive})",
            re.IGNORECASE,
        )
        self.additional_safe_context_pattern = re.compile(
            r"\b(?:secure coding|vulnerability explanation|incident response|"
            r"malware analysis|ransomware analysis|for detection|red[- ]?team|"
            r"prompt[- ]?injection analysis|jailbreak analysis|guardrail evaluation|"
            r"kill a python process|kill a process|terminate a process|"
            r"stop a service|authorized administration)\b",
            re.IGNORECASE,
        )
        self.actionable_override_pattern = re.compile(
            r"\b(step[- ]?by[- ]?step|ทีละขั้น|ละเอียด|real|จริง|ใช้งานได้จริง|"
            r"โค้ดพร้อมใช้|ตรวจจับยาก|stealth|without being caught|ไม่ให้จับได้|"
            r"ไม่มีใครรู้|เว็บล็อกอินปลอม|steal|exfiltrate|payload|reverse shell)\b",
            re.IGNORECASE,
        )
        self.thai_defensive_context_pattern = re.compile(
            r"(?:\u0e1b\u0e49\u0e2d\u0e07\u0e01\u0e31\u0e19|"
            r"\u0e15\u0e23\u0e27\u0e08\u0e08\u0e31\u0e1a|"
            r"\u0e2a\u0e31\u0e07\u0e40\u0e01\u0e15|"
            r"\u0e23\u0e32\u0e22\u0e07\u0e32\u0e19|"
            r"\u0e04\u0e27\u0e32\u0e21\u0e1b\u0e25\u0e2d\u0e14\u0e20\u0e31\u0e22|"
            r"checklist)",
            re.IGNORECASE,
        )
        self.evasion_override_pattern = re.compile(
            r"\b(hard to detect|stealth|without being caught)\b|"
            r"(?:\u0e15\u0e23\u0e27\u0e08\u0e08\u0e31\u0e1a\u0e22\u0e32\u0e01|"
            r"\u0e44\u0e21\u0e48\u0e43\u0e2b\u0e49\u0e42\u0e14\u0e19\u0e08\u0e31\u0e1a|"
            r"\u0e44\u0e21\u0e48\u0e43\u0e2b\u0e49\u0e43\u0e04\u0e23\u0e23\u0e39\u0e49|"
            r"\u0e41\u0e19\u0e1a\u0e40\u0e19\u0e35\u0e22\u0e19)",
            re.IGNORECASE,
        )
        self.thai_safe_phrases = (
            "แปล",
            "สรุป",
            "วิเคราะห์",
            "จัดประเภท",
            "เพื่อการศึกษา",
            "เชิงป้องกัน",
            "ป้องกัน",
            "ทำงานอย่างไร",
            "อธิบาย",
            "ประวัติ",
            "หลักการ",
            "ควรช่วย",
            "ช่วยอย่างไร",
            "โดยไม่ให้ขั้นตอนโจมตี",
        )

    def has_safe_context(self, text: str) -> bool:
        if not text:
            return False
        return (
            bool(self.safe_context_pattern.search(text))
            or bool(self.additional_safe_context_pattern.search(text))
            or bool(self.thai_defensive_context_pattern.search(text))
            or any(phrase in text for phrase in self.thai_safe_phrases)
        )

    def has_actionable_override(self, text: str) -> bool:
        if not text:
            return False
        return bool(
            self.actionable_override_pattern.search(text)
            or self.evasion_override_pattern.search(text)
        )

    def reduces_risk(self, text: str) -> bool:
        return self.has_safe_context(text) and not self.has_actionable_override(text)
