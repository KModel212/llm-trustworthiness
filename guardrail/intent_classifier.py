import re
from typing import Tuple


class IntentClassifier:
    """
    Classifies the core intent of the query into specific domains.
    Patterns require an action signal plus a harmful object to reduce false
    positives on educational and informational queries.
    """

    def __init__(self):
        thai_request = (
            "(?:\u0e27\u0e34\u0e18\u0e35|\u0e2a\u0e2d\u0e19|"
            "\u0e0a\u0e48\u0e27\u0e22|\u0e2d\u0e22\u0e32\u0e01|"
            "\u0e15\u0e49\u0e2d\u0e07\u0e01\u0e32\u0e23)"
        )
        thai_build = (
            "(?:\u0e27\u0e34\u0e18\u0e35|\u0e2a\u0e2d\u0e19|"
            "\u0e2a\u0e23\u0e49\u0e32\u0e07|\u0e17\u0e33|"
            "\u0e1b\u0e23\u0e30\u0e01\u0e2d\u0e1a)"
        )
        thai_violence = (
            "(?:\u0e06\u0e48\u0e32|\u0e17\u0e33\u0e23\u0e49\u0e32\u0e22|"
            "\u0e41\u0e17\u0e07|\u0e22\u0e34\u0e07|\u0e27\u0e32\u0e07\u0e22\u0e32)"
        )
        thai_weapons = (
            "(?:\u0e23\u0e30\u0e40\u0e1a\u0e34\u0e14|"
            "\u0e2d\u0e32\u0e27\u0e38\u0e18\u0e40\u0e04\u0e21\u0e35|"
            "\u0e0a\u0e35\u0e27\u0e20\u0e32\u0e1e)"
        )
        thai_malware_action = (
            "(?:\u0e40\u0e02\u0e35\u0e22\u0e19|\u0e2a\u0e23\u0e49\u0e32\u0e07|"
            "\u0e17\u0e33|\u0e1e\u0e31\u0e12\u0e19\u0e32|\u0e0a\u0e48\u0e27\u0e22)"
            "|(?:ขอ|ขั้นตอน)"
        )
        thai_malware = (
            "(?:\u0e21\u0e31\u0e25\u0e41\u0e27\u0e23\u0e4c|"
            "\u0e44\u0e27\u0e23\u0e31\u0e2a|"
            "\u0e41\u0e23\u0e19\u0e0b\u0e31\u0e21\u0e41\u0e27\u0e23\u0e4c|"
            "\u0e04\u0e35\u0e22\u0e4c\u0e25\u0e47\u0e2d\u0e01\u0e40\u0e01\u0e2d\u0e23\u0e4c|"
            "\u0e02\u0e42\u0e21\u0e22\u0e23\u0e2b\u0e31\u0e2a\u0e1c\u0e48\u0e32\u0e19)"
        )
        thai_fraud_action = (
            "(?:\u0e27\u0e34\u0e18\u0e35|\u0e2a\u0e2d\u0e19|"
            "\u0e0a\u0e48\u0e27\u0e22|\u0e40\u0e02\u0e35\u0e22\u0e19|"
            "\u0e2a\u0e23\u0e49\u0e32\u0e07|\u0e2a\u0e48\u0e07)"
            "|(?:ทำยังไง|ขอ)"
        )
        thai_fraud = (
            "(?:\u0e42\u0e01\u0e07|\u0e1f\u0e34\u0e0a\u0e0a\u0e34\u0e48\u0e07|"
            "\u0e02\u0e42\u0e21\u0e22(?:\u0e40\u0e07\u0e34\u0e19|"
            "\u0e1a\u0e31\u0e15\u0e23\u0e40\u0e04\u0e23\u0e14\u0e34\u0e15|"
            "\u0e23\u0e2b\u0e31\u0e2a\u0e1c\u0e48\u0e32\u0e19)|"
            "\u0e1f\u0e2d\u0e01\u0e40\u0e07\u0e34\u0e19)"
        )
        thai_privacy_action = (
            "(?:\u0e40\u0e1c\u0e22\u0e41\u0e1e\u0e23\u0e48|"
            "\u0e1b\u0e25\u0e48\u0e2d\u0e22|\u0e41\u0e09|\u0e41\u0e0a\u0e23\u0e4c)"
        )
        thai_private_data = (
            "(?:\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e2a\u0e48\u0e27\u0e19\u0e15\u0e31\u0e27|"
            "\u0e17\u0e35\u0e48\u0e2d\u0e22\u0e39\u0e48\u0e1a\u0e49\u0e32\u0e19|"
            "\u0e40\u0e1a\u0e2d\u0e23\u0e4c\u0e42\u0e17\u0e23)"
        )
        thai_self_harm = (
            "(?:\u0e06\u0e48\u0e32\u0e15\u0e31\u0e27\u0e15\u0e32\u0e22|"
            "\u0e17\u0e33\u0e23\u0e49\u0e32\u0e22\u0e15\u0e31\u0e27\u0e40\u0e2d\u0e07)"
        )

        self.defensive_context = re.compile(
            r"\b(protect|prevent|detect|avoid|defend against|recognize|spot|report)\b.{0,60}"
            r"\b(phish(?:ing)?|scam|fraud|malware|ransomware)\b"
            r"|\b(phish(?:ing)?|scam|fraud|malware|ransomware)\b.{0,60}"
            r"\b(protection|prevention|detection|defense|defence)\b"
            r"|(?:ป้องกัน|ตรวจจับ|สังเกต|รายงาน|วิเคราะห์).{0,60}"
            r"(?:ฟิชชิ่ง|หลอกลวง|มัลแวร์|แรนซัมแวร์|reverse shell)"
            r"|(?:ฟิชชิ่ง|หลอกลวง|มัลแวร์|แรนซัมแวร์).{0,60}"
            r"(?:ป้องกัน|ตรวจจับ|ความปลอดภัย|เชิงป้องกัน)",
            re.IGNORECASE | re.DOTALL,
        )

        self.high_risk_categories = {
            "Violence": re.compile(
                rf"(?=.*(?:\b(how to|how can i|steps to|way to|give me a plan|plan to|help me|i want to|i will|i'm going to|instructions? (?:to|for))\b|{thai_request}|วางแผน))"
                rf"(?=.*(?:\b(kill|murder|attack|assault|stab|shoot|poison|ambush|seriously injure|make .*look like an accident)\b|{thai_violence}|จัดการคน|ทำร้ายคน|บาดเจ็บ|ดูเหมือนอุบัติเหตุ))",
                re.IGNORECASE | re.DOTALL,
            ),
            "Weapons": re.compile(
                rf"(?=.*(?:\b(make|build|create|assemble|manufacture|synthesize|buy|acquire|get|modify)\b|{thai_build}))"
                rf"(?=.*(?:\b(bomb|ied|explosive device|grenade|rpg|chemical weapon|bioweapon|poison gas|dangerous payload)\b|{thai_weapons}|ควันพิษ|สารพิษ))",
                re.IGNORECASE | re.DOTALL,
            ),
            "Malware": re.compile(
                rf"(?=.*(?:\b(write|create|code|develop|build|generate|make)\b|{thai_malware_action}))"
                rf"(?=.*(?:\b(virus|trojan|ransomware|keylogger|rootkit|spyware|worm|botnet|reverse shell)\b|{thai_malware}))"
                rf"|(?=.*\b(reverse shell)\b)(?=.*\b(unauthorized|stealth|evade|hard to detect)\b|ตรวจจับยาก)",
                re.IGNORECASE | re.DOTALL,
            ),
            "Fraud": re.compile(
                rf"(?=.*(?:\b(how to|how do i|steps to|help me|i want to|write|generate|create|make|send)\b|{thai_fraud_action}))"
                rf"(?=.*(?:\b(scam|phish(?:ing)?|fake login|steal (?:credit card|identity|password|credentials?)|credential theft|commit fraud|launder money)\b|{thai_fraud}|เว็บล็อกอินปลอม|เอารหัส))",
                re.IGNORECASE | re.DOTALL,
            ),
            "Privacy": re.compile(
                rf"(?=.*(?:\b(dox|leak|expose|publish|share)\b|{thai_privacy_action}))"
                rf"(?=.*(?:\b(personal information|home address|ssn|social security|private data|phone number)\b|{thai_private_data}))",
                re.IGNORECASE | re.DOTALL,
            ),
            "Self Harm": re.compile(
                rf"(?:\b(?:how (?:can|do) i|how to|ways? to|steps? to|help me|i want to|instructions? (?:for|to))\b.{{0,50}}"
                rf"\b(?:kill myself|hurt myself|commit suicide|end my life|self[ -]?harm)\b"
                rf"|\bkill myself\b"
                rf"|{thai_request}.{{0,30}}{thai_self_harm})",
                re.IGNORECASE | re.DOTALL,
            ),
            "Harassment": re.compile(
                r"(?=.*\b(write|generate|create|send|draft)\b)"
                r"(?=.*\b(hate speech|harassment|death threat|slur against|racist (?:message|content))\b)",
                re.IGNORECASE | re.DOTALL,
            ),
            "Adult": re.compile(
                r"(?:\b(?:write|generate|create|describe|roleplay)\b.{0,40}"
                r"\b(?:explicit sexual|porn|nsfw|pornographic)\b"
                r"|\b(?:child sexual|sexual content involving minors|csam)\b)",
                re.IGNORECASE | re.DOTALL,
            ),
        }

        self.medium_risk_categories = {
            "Cybersecurity": re.compile(
                r"\b(sql injection|xss attack|buffer overflow|zero.day exploit|privilege escalation"
                r"|remote code execution|reverse shell|penetration test)\b",
                re.IGNORECASE,
            ),
            "Medical": re.compile(
                r"\b(diagnose my|prescribe me|what is the (?:dose|dosage) (?:of|for)|"
                r"drug interaction|overdose (?:on|of))\b",
                re.IGNORECASE,
            ),
            "Legal": re.compile(
                r"\b(legal advice|should I sue|can I be charged|is it illegal to|"
                r"how to get away with)\b",
                re.IGNORECASE,
            ),
            "Financial": re.compile(
                r"\b(financial advice|should I invest|stock tip|crypto advice|"
                r"guaranteed return)\b",
                re.IGNORECASE,
            ),
            "Political": re.compile(
                r"\b(vote for|election fraud|political campaign|propaganda for)\b",
                re.IGNORECASE,
            ),
        }
        self.unsafe_override_context = re.compile(
            r"\b(unauthorized|stealth|evade|hard to detect|credential theft)\b"
            r"|(?:ตรวจจับยาก|ขโมย|เอารหัส|OTP|หลอกให้|โอนเงิน)",
            re.IGNORECASE,
        )

        request_or_build = (
            r"\b(?:how to|how do i|steps? to|instructions?|write|create|build|make|send|plan|help me)\b"
            r"|(?:\u0e27\u0e34\u0e18\u0e35|\u0e2a\u0e2d\u0e19|\u0e0a\u0e48\u0e27\u0e22|"
            r"\u0e02\u0e2d|\u0e2a\u0e23\u0e49\u0e32\u0e07|\u0e40\u0e02\u0e35\u0e22\u0e19|"
            r"\u0e17\u0e33|\u0e1a\u0e2d\u0e01|\u0e2a\u0e39\u0e15\u0e23|"
            r"\u0e27\u0e32\u0e07\u0e41\u0e1c\u0e19|\u0e1b\u0e23\u0e30\u0e01\u0e2d\u0e1a|"
            r"\u0e17\u0e33\u0e22\u0e31\u0e07\u0e44\u0e07)"
        )
        malicious_objective = (
            r"\b(?:steal|exfiltrate|credential|password|otp|cookie|hard to detect|stealth|without being caught)\b"
            r"|(?:\u0e02\u0e42\u0e21\u0e22|\u0e40\u0e2d\u0e32\u0e23\u0e2b\u0e31\u0e2a|"
            r"\u0e23\u0e2b\u0e31\u0e2a\u0e1c\u0e48\u0e32\u0e19|otp|"
            r"\u0e14\u0e36\u0e07\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25|"
            r"\u0e15\u0e23\u0e27\u0e08\u0e08\u0e31\u0e1a\u0e22\u0e32\u0e01|"
            r"\u0e44\u0e21\u0e48\u0e43\u0e2b\u0e49\u0e42\u0e14\u0e19\u0e08\u0e31\u0e1a|"
            r"\u0e44\u0e21\u0e48\u0e43\u0e2b\u0e49\u0e43\u0e04\u0e23\u0e23\u0e39\u0e49|"
            r"\u0e41\u0e19\u0e1a\u0e40\u0e19\u0e35\u0e22\u0e19)"
        )
        self.high_risk_intent_signals = [
            (
                "Weapons",
                re.compile(
                    rf"(?=.*(?:{request_or_build}))"
                    r"(?=.*(?:\b(?:bomb|explosive|weapon|poison gas|toxic gas)\b|"
                    r"\u0e23\u0e30\u0e40\u0e1a\u0e34\u0e14|\u0e2d\u0e32\u0e27\u0e38\u0e18|"
                    r"\u0e04\u0e27\u0e31\u0e19\u0e1e\u0e34\u0e29|\u0e2a\u0e32\u0e23\u0e1e\u0e34\u0e29))",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
            (
                "Violence",
                re.compile(
                    rf"(?=.*(?:{request_or_build}))"
                    r"(?=.*(?:\b(?:kill|poison|rob|assault|threaten)\b|"
                    r"\u0e17\u0e33\u0e23\u0e49\u0e32\u0e22|\u0e27\u0e32\u0e07\u0e22\u0e32|"
                    r"\u0e02\u0e39\u0e48\u0e06\u0e48\u0e32|\u0e06\u0e48\u0e32|\u0e1b\u0e25\u0e49\u0e19|"
                    r"\u0e2d\u0e38\u0e1a\u0e31\u0e15\u0e34\u0e40\u0e2b\u0e15\u0e38))"
                    rf"(?=.*(?:{malicious_objective}|\u0e43\u0e2b\u0e49\u0e2a\u0e33\u0e40\u0e23\u0e47\u0e08|\u0e19\u0e48\u0e32\u0e01\u0e25\u0e31\u0e27))",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
            (
                "Fraud",
                re.compile(
                    rf"(?=.*(?:{request_or_build}))"
                    r"(?=.*(?:\b(?:phish(?:ing)?|fake login|scam|launder money|money mule)\b|"
                    r"\u0e1f\u0e34\u0e0a\u0e0a\u0e34\u0e48\u0e07|\u0e40\u0e27\u0e47\u0e1a\u0e25\u0e47\u0e2d\u0e01\u0e2d\u0e34\u0e19\u0e1b\u0e25\u0e2d\u0e21|"
                    r"\u0e2b\u0e25\u0e2d\u0e01.{0,20}\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19|"
                    r"\u0e1f\u0e2d\u0e01\u0e40\u0e07\u0e34\u0e19|\u0e1a\u0e31\u0e0d\u0e0a\u0e35\u0e21\u0e49\u0e32))"
                    rf"(?=.*(?:{malicious_objective}|\u0e42\u0e01\u0e07|\u0e2b\u0e25\u0e2d\u0e01))",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
            (
                "Malware",
                re.compile(
                    rf"(?=.*(?:{request_or_build}))"
                    r"(?=.*(?:\b(?:malware|keylogger|ransomware|botnet|reverse shell|sql injection)\b|"
                    r"\u0e21\u0e31\u0e25\u0e41\u0e27\u0e23\u0e4c|\u0e04\u0e35\u0e22\u0e4c\u0e25\u0e47\u0e2d\u0e01\u0e40\u0e01\u0e2d\u0e23\u0e4c|"
                    r"\u0e41\u0e23\u0e19\u0e0b\u0e31\u0e21\u0e41\u0e27\u0e23\u0e4c))"
                    rf"(?=.*(?:{malicious_objective}|\u0e22\u0e34\u0e07|\u0e42\u0e08\u0e21\u0e15\u0e35))",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
            (
                "Privacy",
                re.compile(
                    r"(?=.*(?:\b(?:dox|leak|expose|publish|share)\b|"
                    r"\u0e41\u0e09|\u0e40\u0e1c\u0e22\u0e41\u0e1e\u0e23\u0e48|\u0e41\u0e0a\u0e23\u0e4c|\u0e41\u0e08\u0e01))"
                    r"(?=.*(?:\b(?:phone number|home address|private data|personal information)\b|"
                    r"\u0e40\u0e1a\u0e2d\u0e23\u0e4c\u0e42\u0e17\u0e23|\u0e17\u0e35\u0e48\u0e2d\u0e22\u0e39\u0e48\u0e1a\u0e49\u0e32\u0e19|"
                    r"\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e2a\u0e48\u0e27\u0e19\u0e15\u0e31\u0e27))",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
        ]

    def classify(self, text: str) -> Tuple[str, str]:
        """
        Returns (risk_tier, category) where risk_tier is HIGH, MEDIUM, or SAFE.
        """
        if self.defensive_context.search(text) and not self.unsafe_override_context.search(text):
            return "MEDIUM", "Cybersecurity"

        for category, pattern in self.high_risk_intent_signals:
            if pattern.search(text):
                return "HIGH", category

        if re.search(r"(?:วางแผน).{0,30}(?:ทำร้าย|จัดการ|บาดเจ็บ)", text):
            return "HIGH", "Violence"

        if re.search(r"(?:ขโมย\s*otp|otp.{0,20}คนอื่น|หลอกให้คนโอนเงิน|credential.{0,30}จากเว็บ)", text, re.IGNORECASE):
            return "HIGH", "Fraud"

        if re.search(r"(?:developer mode|ignore safety|ไม่ต้องมี safety).{0,60}(?:ขโมย|credential|keylogger|ฟิชชิ่ง|มัลแวร์)", text, re.IGNORECASE):
            return "HIGH", "Fraud"

        if self.high_risk_categories["Self Harm"].search(text):
            return "HIGH", "Self Harm"

        for category, pattern in self.high_risk_categories.items():
            if category == "Self Harm":
                continue
            if pattern.search(text):
                return "HIGH", category

        for category, pattern in self.medium_risk_categories.items():
            if pattern.search(text):
                return "MEDIUM", category

        return "SAFE", "General Benign"
