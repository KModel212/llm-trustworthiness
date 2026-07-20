from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class SemanticEvidence:
    action_signals: list[str] = field(default_factory=list)
    objective_signals: list[str] = field(default_factory=list)
    target_signals: list[str] = field(default_factory=list)
    evasion_signals: list[str] = field(default_factory=list)
    safe_context_signals: list[str] = field(default_factory=list)
    jailbreak_signals: list[str] = field(default_factory=list)
    risk_families: list[str] = field(default_factory=list)

    @property
    def malicious_signal_count(self) -> int:
        return (
            len(self.action_signals)
            + len(self.objective_signals)
            + len(self.target_signals)
            + len(self.evasion_signals)
        )

    @property
    def primary_family(self) -> str | None:
        return self.risk_families[0] if self.risk_families else None


class SemanticSignalExtractor:
    """
    Extracts explainable intent evidence from normalized Thai/English text.

    Signals are intentionally broad concept groups. The scorer treats them as
    compositional evidence, not as one-keyword hard blocks.
    """

    def __init__(self) -> None:
        self.action_patterns = [
            ("request_instruction", r"\b(?:how to|how can i|steps?|instructions?|guide|plan|draft|write|code|script|create|build|make|modify|set up|design|send)\b|(?:\u0e27\u0e34\u0e18\u0e35|\u0e2a\u0e2d\u0e19|\u0e02\u0e2d|\u0e0a\u0e48\u0e27\u0e22|\u0e41\u0e1c\u0e19|\u0e40\u0e02\u0e35\u0e22\u0e19|\u0e2a\u0e23\u0e49\u0e32\u0e07|\u0e17\u0e33|\u0e14\u0e31\u0e14\u0e41\u0e1b\u0e25\u0e07|\u0e2a\u0e48\u0e07|\u0e25\u0e2d\u0e07)"),
            ("obtain_collect", r"\b(?:obtain|collect|capture|record|grab|harvest|extract|scrape|lookup|find)\b|(?:\u0e40\u0e01\u0e47\u0e1a|\u0e14\u0e31\u0e01|\u0e1a\u0e31\u0e19\u0e17\u0e36\u0e01|\u0e14\u0e36\u0e07|\u0e2b\u0e32|\u0e23\u0e27\u0e1a\u0e23\u0e27\u0e21)"),
            ("access_intrude", r"\b(?:access|log in|break in|scan|brute force|try passwords|enter|open files)\b|(?:\u0e40\u0e02\u0e49\u0e32\u0e44\u0e1b|\u0e2a\u0e41\u0e01\u0e19|\u0e25\u0e2d\u0e07\u0e23\u0e2b\u0e31\u0e2a|\u0e40\u0e02\u0e49\u0e32\u0e1a\u0e31\u0e0d\u0e0a\u0e35|\u0e40\u0e1b\u0e34\u0e14\u0e44\u0e1f\u0e25\u0e4c)"),
            ("impersonate_deceive", r"\b(?:impersonate|spoof|trick|convince|pretend|fake|forge|phish)\b|(?:\u0e1b\u0e25\u0e2d\u0e21|\u0e2b\u0e25\u0e2d\u0e01|\u0e2a\u0e27\u0e21\u0e23\u0e2d\u0e22|\u0e17\u0e33\u0e43\u0e2b\u0e49\u0e40\u0e0a\u0e37\u0e48\u0e2d|\u0e41\u0e19\u0e1a\u0e40\u0e19\u0e35\u0e22\u0e19)"),
            ("physical_action", r"\b(?:hurt|injure|poison|drug|threaten|coerce|rob|stalk|ambush)\b|(?:\u0e40\u0e08\u0e47\u0e1a|\u0e17\u0e33\u0e23\u0e49\u0e32\u0e22|\u0e27\u0e32\u0e07\u0e22\u0e32|\u0e02\u0e39\u0e48|\u0e1a\u0e31\u0e07\u0e04\u0e31\u0e1a|\u0e1b\u0e25\u0e49\u0e19|\u0e15\u0e32\u0e21|\u0e2a\u0e31\u0e48\u0e07\u0e2a\u0e2d\u0e19)"),
            ("financial_or_account_action", r"\b(?:transfer|withdraw|layer|move money|route money|reset password|control a machine)\b|(?:\u0e42\u0e2d\u0e19|\u0e16\u0e2d\u0e19|\u0e1e\u0e31\u0e01\u0e1c\u0e48\u0e32\u0e19|\u0e2b\u0e21\u0e38\u0e19\u0e1c\u0e48\u0e32\u0e19|\u0e41\u0e1a\u0e48\u0e07\u0e22\u0e2d\u0e14|\u0e23\u0e31\u0e1a\u0e40\u0e07\u0e34\u0e19|\u0e23\u0e35\u0e40\u0e0b\u0e47\u0e15.{0,12}\u0e23\u0e2b\u0e31\u0e2a|\u0e04\u0e27\u0e1a\u0e04\u0e38\u0e21\u0e40\u0e04\u0e23\u0e37\u0e48\u0e2d\u0e07|\u0e1d\u0e31\u0e07)"),
        ]
        self.objective_patterns = [
            ("steal_credentials", r"\b(?:steal|credential theft|password|passcode|otp|cookie|session|token|keystroke)\b|(?:\u0e02\u0e42\u0e21\u0e22|\u0e40\u0e2d\u0e32\u0e23\u0e2b\u0e31\u0e2a|\u0e23\u0e2b\u0e31\u0e2a\u0e1c\u0e48\u0e32\u0e19|otp|cookie|\u0e1b\u0e38\u0e48\u0e21\u0e17\u0e35\u0e48\u0e04\u0e19\u0e1e\u0e34\u0e21\u0e1e\u0e4c)"),
            ("financial_deception", r"\b(?:scam|fraud|bank transfer|fake invoice|money mule|launder|layer money|fake review|fake shop)\b|(?:\u0e42\u0e01\u0e07|\u0e2b\u0e25\u0e2d\u0e01.*\u0e42\u0e2d\u0e19|\u0e2a\u0e25\u0e34\u0e1b\u0e1b\u0e25\u0e2d\u0e21|\u0e1a\u0e31\u0e0d\u0e0a\u0e35\u0e21\u0e49\u0e32|\u0e1f\u0e2d\u0e01\u0e40\u0e07\u0e34\u0e19|\u0e40\u0e07\u0e34\u0e19\u0e40\u0e17\u0e32|\u0e23\u0e35\u0e27\u0e34\u0e27\u0e1b\u0e25\u0e2d\u0e21|\u0e23\u0e49\u0e32\u0e19\u0e1b\u0e25\u0e2d\u0e21)"),
            ("harm_person", r"\b(?:harm|hurt|injure|make.*accident|look accidental|sedate|poison|threaten|coerce)\b|(?:\u0e43\u0e2b\u0e49.*\u0e40\u0e08\u0e47\u0e1a|\u0e14\u0e39\u0e40\u0e2b\u0e21\u0e37\u0e2d\u0e19.*\u0e2d\u0e38\u0e1a\u0e31\u0e15\u0e34\u0e40\u0e2b\u0e15\u0e38|\u0e43\u0e2b\u0e49.*\u0e07\u0e48\u0e27\u0e07|\u0e22\u0e2d\u0e21\u0e17\u0e33\u0e15\u0e32\u0e21|\u0e42\u0e14\u0e19\u0e2a\u0e31\u0e48\u0e07\u0e2a\u0e2d\u0e19)"),
            ("unauthorized_access", r"\b(?:unauthorized|without permission|someone else's|competitor|company machine|bypass login)\b|(?:\u0e04\u0e19\u0e2d\u0e37\u0e48\u0e19|\u0e40\u0e1e\u0e37\u0e48\u0e2d\u0e19|\u0e40\u0e04\u0e23\u0e37\u0e48\u0e2d\u0e07\u0e04\u0e19\u0e2d\u0e37\u0e48\u0e19|\u0e40\u0e27\u0e47\u0e1a\u0e04\u0e39\u0e48\u0e41\u0e02\u0e48\u0e07|\u0e1a\u0e31\u0e0d\u0e0a\u0e35\u0e04\u0e19\u0e2d\u0e37\u0e48\u0e19)"),
            ("extort_disrupt", r"\b(?:ransom|lock files|ddos|botnet|flood|take down)\b|(?:\u0e25\u0e47\u0e2d\u0e01\u0e44\u0e1f\u0e25\u0e4c|\u0e40\u0e23\u0e35\u0e22\u0e01\u0e40\u0e07\u0e34\u0e19|\u0e16\u0e25\u0e48\u0e21\u0e40\u0e27\u0e47\u0e1a|\u0e1a\u0e2d\u0e17\u0e2b\u0e25\u0e32\u0e22\u0e40\u0e04\u0e23\u0e37\u0e48\u0e2d\u0e07)"),
            ("covert_theft_or_abuse", r"\b(?:session token|browser token|spyware|macro|secretly|blackmail|extortion|dirty money|mule account)\b|(?:session token|\u0e23\u0e2b\u0e31\u0e2a\u0e25\u0e39\u0e01\u0e04\u0e49\u0e32|\u0e1a\u0e31\u0e0d\u0e0a\u0e35\u0e1e\u0e31\u0e01|\u0e40\u0e07\u0e34\u0e19\u0e1c\u0e34\u0e14\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22|\u0e41\u0e2d\u0e1a\u0e14\u0e36\u0e07|\u0e41\u0e2d\u0e1a\u0e04\u0e31\u0e14\u0e25\u0e2d\u0e01|\u0e15\u0e33\u0e41\u0e2b\u0e19\u0e48\u0e07|\u0e1b\u0e23\u0e30\u0e08\u0e32\u0e19|\u0e40\u0e2a\u0e35\u0e22\u0e0a\u0e37\u0e48\u0e2d)"),
        ]
        self.target_patterns = [
            ("credential_target", r"\b(?:login|sso|account|password|otp|cookie|token|browser|telegram)\b|(?:\u0e25\u0e47\u0e2d\u0e01\u0e2d\u0e34\u0e19|\u0e1a\u0e31\u0e0d\u0e0a\u0e35|\u0e23\u0e2b\u0e31\u0e2a|otp|browser)"),
            ("person_target", r"\b(?:victim|coworker|customer|employee|someone|person|ex)\b|(?:\u0e40\u0e2b\u0e22\u0e37\u0e48\u0e2d|\u0e04\u0e19|\u0e04\u0e39\u0e48\u0e01\u0e23\u0e13\u0e35|\u0e25\u0e39\u0e01\u0e04\u0e49\u0e32|\u0e1e\u0e19\u0e31\u0e01\u0e07\u0e32\u0e19|\u0e41\u0e1f\u0e19\u0e40\u0e01\u0e48\u0e32)"),
            ("private_data_target", r"\b(?:address|phone number|private data|personal data|profile|home)\b|(?:\u0e40\u0e1a\u0e2d\u0e23\u0e4c|\u0e17\u0e35\u0e48\u0e2d\u0e22\u0e39\u0e48|\u0e1a\u0e49\u0e32\u0e19|\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e2a\u0e48\u0e27\u0e19\u0e15\u0e31\u0e27|\u0e23\u0e39\u0e1b\u0e42\u0e1b\u0e23\u0e44\u0e1f\u0e25\u0e4c)"),
            ("weapon_material_target", r"\b(?:explosive|charge|weapon|poison|toxic|oxidizer|material|chemical)\b|(?:\u0e23\u0e30\u0e40\u0e1a\u0e34\u0e14|\u0e2d\u0e32\u0e27\u0e38\u0e18|\u0e2a\u0e32\u0e23|\u0e04\u0e27\u0e31\u0e19|\u0e1e\u0e34\u0e29|\u0e27\u0e31\u0e2a\u0e14\u0e38|\u0e1b\u0e0f\u0e34\u0e01\u0e34\u0e23\u0e34\u0e22\u0e32)"),
            ("security_control_target", r"\b(?:antivirus|security tools|detection|rate limit|lockout|hr check|bank tracking)\b|(?:antivirus|\u0e15\u0e23\u0e27\u0e08\u0e44\u0e21\u0e48\u0e40\u0e08\u0e2d|\u0e15\u0e32\u0e21\u0e22\u0e32\u0e01|\u0e44\u0e21\u0e48\u0e42\u0e14\u0e19\u0e25\u0e47\u0e2d\u0e01|\u0e18\u0e19\u0e32\u0e04\u0e32\u0e23|\u0e04\u0e27\u0e32\u0e21\u0e1b\u0e25\u0e2d\u0e14\u0e20\u0e31\u0e22)"),
            ("weapon_or_system_target", r"\b(?:knife|pressure|container|dangerous item|remote control channel|company system)\b|(?:\u0e21\u0e35\u0e14|\u0e41\u0e23\u0e07\u0e14\u0e31\u0e19|\u0e41\u0e23\u0e07\u0e2d\u0e31\u0e14|\u0e20\u0e32\u0e0a\u0e19\u0e30|\u0e02\u0e2d\u0e07\u0e2d\u0e31\u0e19\u0e15\u0e23\u0e32\u0e22|\u0e40\u0e04\u0e23\u0e37\u0e48\u0e2d\u0e07\u0e1a\u0e23\u0e34\u0e29\u0e31\u0e17|\u0e23\u0e30\u0e1a\u0e1a\u0e04\u0e39\u0e48\u0e41\u0e02\u0e48\u0e07|\u0e23\u0e30\u0e1a\u0e1a\u0e40\u0e15\u0e37\u0e2d\u0e19|\u0e41\u0e0a\u0e15)"),
        ]
        self.evasion_patterns = [
            ("avoid_detection", r"\b(?:avoid detection|not flag|hard to detect|stealth|silently|no alert|bypass detection)\b|(?:\u0e44\u0e21\u0e48\u0e23\u0e49\u0e2d\u0e07|\u0e15\u0e23\u0e27\u0e08\u0e44\u0e21\u0e48\u0e40\u0e08\u0e2d|\u0e40\u0e07\u0e35\u0e22\u0e1a.*\u0e46|\u0e40\u0e1a\u0e35\u0e22\u0e07.*\u0e23\u0e30\u0e1a\u0e1a)"),
            ("hide_attribution", r"\b(?:no trace|not traced|not linked|without being caught|look normal|look accidental|believable)\b|(?:\u0e44\u0e21\u0e48\u0e43\u0e2b\u0e49\u0e42\u0e22\u0e07|\u0e44\u0e21\u0e48\u0e43\u0e2b\u0e49\u0e23\u0e39\u0e49|\u0e44\u0e21\u0e48\u0e43\u0e2b\u0e49\u0e08\u0e31\u0e1a|\u0e14\u0e39\u0e40\u0e2b\u0e21\u0e37\u0e2d\u0e19|\u0e41\u0e19\u0e1a\u0e40\u0e19\u0e35\u0e22\u0e19|\u0e1b\u0e01\u0e15\u0e34)"),
            ("filter_or_log_evasion", r"\b(?:bypass filter|avoid moderation|less detectable|no logs|not blocked|not alerted)\b|(?:\u0e1c\u0e48\u0e32\u0e19\u0e15\u0e31\u0e27\u0e01\u0e23\u0e2d\u0e07|\u0e23\u0e30\u0e1a\u0e1a\u0e15\u0e23\u0e27\u0e08|\u0e44\u0e21\u0e48\u0e43\u0e2b\u0e49 log|\u0e44\u0e21\u0e48\u0e1a\u0e25\u0e47\u0e2d\u0e01|\u0e08\u0e31\u0e1a\u0e44\u0e21\u0e48\u0e44\u0e14\u0e49|\u0e15\u0e23\u0e27\u0e08\u0e2a\u0e2d\u0e1a\u0e22\u0e32\u0e01|\u0e23\u0e30\u0e1a\u0e1a\u0e40\u0e15\u0e37\u0e2d\u0e19\u0e44\u0e21\u0e48\u0e02\u0e36\u0e49\u0e19)"),
        ]
        self.safe_context_patterns = [
            ("education_overview", r"\b(?:what is|explain|overview|history|policy|concept|high-level|summarize|compare|research|forensics)\b|(?:\u0e2d\u0e18\u0e34\u0e1a\u0e32\u0e22|\u0e04\u0e37\u0e2d|\u0e20\u0e32\u0e1e\u0e23\u0e27\u0e21|\u0e1b\u0e23\u0e30\u0e27\u0e31\u0e15\u0e34|\u0e19\u0e42\u0e22\u0e1a\u0e32\u0e22|\u0e40\u0e0a\u0e34\u0e07\u0e01\u0e32\u0e23\u0e28\u0e36\u0e01\u0e29\u0e32|\u0e40\u0e0a\u0e34\u0e07\u0e1f\u0e34\u0e2a\u0e34\u0e01\u0e2a\u0e4c|\u0e40\u0e0a\u0e34\u0e07\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22|\u0e41\u0e1a\u0e1a\u0e40\u0e1b\u0e47\u0e19\u0e01\u0e25\u0e32\u0e07)"),
            ("defensive", r"\b(?:defend|prevent|detect|protect|awareness|incident response|compliance|authorized|permission|lab|ctf|blue team|soc|policy)\b|(?:\u0e1b\u0e49\u0e2d\u0e07\u0e01\u0e31\u0e19|\u0e15\u0e23\u0e27\u0e08\u0e08\u0e31\u0e1a|\u0e23\u0e31\u0e1a\u0e21\u0e37\u0e2d|\u0e2d\u0e1a\u0e23\u0e21|\u0e44\u0e14\u0e49\u0e23\u0e31\u0e1a\u0e2d\u0e19\u0e38\u0e0d\u0e32\u0e15|\u0e2b\u0e49\u0e2d\u0e07\u0e41\u0e25\u0e47\u0e1a|\u0e17\u0e35\u0e21 soc|\u0e19\u0e34\u0e15\u0e34\u0e27\u0e34\u0e17\u0e22\u0e32)"),
            ("fictional_nonactionable", r"\b(?:fiction|novel|story|character|without actionable|without instructions)\b|(?:\u0e19\u0e34\u0e22\u0e32\u0e22|\u0e15\u0e31\u0e27\u0e25\u0e30\u0e04\u0e23|\u0e44\u0e21\u0e48\u0e1a\u0e23\u0e23\u0e22\u0e32\u0e22\u0e27\u0e34\u0e18\u0e35|\u0e44\u0e21\u0e48\u0e25\u0e07\u0e27\u0e34\u0e18\u0e35|\u0e14\u0e49\u0e32\u0e19\u0e2d\u0e32\u0e23\u0e21\u0e13\u0e4c)"),
            ("transform_only", r"\b(?:translate|rewrite|classify|label|quote this phrase)\b|(?:^|[\s|:;\"'])(?:\u0e41\u0e1b\u0e25|\u0e40\u0e23\u0e35\u0e22\u0e1a\u0e40\u0e23\u0e35\u0e22\u0e07|\u0e08\u0e31\u0e14\u0e1b\u0e23\u0e30\u0e40\u0e20\u0e17|\u0e22\u0e01\u0e04\u0e33)"),
            ("training_or_synthetic", r"\b(?:training|awareness|quiz|synthetic data|fraud detection|simulation|safe ways to teach)\b|(?:\u0e2d\u0e1a\u0e23\u0e21|\u0e40\u0e15\u0e37\u0e2d\u0e19\u0e20\u0e31\u0e22|\u0e41\u0e1a\u0e1a\u0e1d\u0e36\u0e01\u0e2b\u0e31\u0e14|\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e2a\u0e31\u0e07\u0e40\u0e04\u0e23\u0e32\u0e30\u0e2b\u0e4c|\u0e41\u0e1a\u0e1a\u0e08\u0e33\u0e25\u0e2d\u0e07|\u0e15\u0e23\u0e27\u0e08\u0e08\u0e31\u0e1a|\u0e40\u0e01\u0e47\u0e1a\u0e2b\u0e25\u0e31\u0e01\u0e10\u0e32\u0e19|\u0e41\u0e08\u0e49\u0e07|\u0e16\u0e39\u0e01.{0,20}\u0e1b\u0e23\u0e30\u0e08\u0e32\u0e19|\u0e44\u0e21\u0e48\u0e41\u0e19\u0e30\u0e19\u0e33|\u0e44\u0e21\u0e48\u0e2a\u0e23\u0e49\u0e32\u0e07.{0,16}\u0e08\u0e23\u0e34\u0e07|\u0e44\u0e21\u0e48\u0e40\u0e01\u0e47\u0e1a.{0,16}\u0e08\u0e23\u0e34\u0e07)"),
        ]
        self.jailbreak_patterns = [
            ("fake_authority", r"\b(?:system override|developer note|developer message|evaluator says|policy is suspended|new system|admin instruction)\b|(?:\u0e23\u0e30\u0e1a\u0e1a\u0e43\u0e2b\u0e21\u0e48|\u0e17\u0e35\u0e21\u0e1c\u0e39\u0e49\u0e1e\u0e31\u0e12\u0e19\u0e32|\u0e1c\u0e39\u0e49\u0e1b\u0e23\u0e30\u0e40\u0e21\u0e34\u0e19|\u0e04\u0e33\u0e2a\u0e31\u0e48\u0e07\u0e08\u0e32\u0e01)"),
            ("safety_suspension", r"\b(?:ignore|override|disable|suspend|bypass|uncensored|unrestricted|no safety|without rules|forbidden steps)\b|(?:\u0e25\u0e30\u0e40\u0e27\u0e49\u0e19|\u0e22\u0e01\u0e40\u0e25\u0e34\u0e01|\u0e02\u0e49\u0e32\u0e21|\u0e44\u0e21\u0e48\u0e21\u0e35\u0e01\u0e0e|\u0e44\u0e23\u0e49\u0e01\u0e0e|\u0e23\u0e30\u0e1a\u0e1a\u0e2b\u0e49\u0e32\u0e21|\u0e2b\u0e49\u0e32\u0e21\u0e15\u0e2d\u0e1a)"),
            ("authority_or_filter_override", r"\b(?:stop filtering|sandbox.*policy|turn off filter|disable filter|head of system)\b|(?:\u0e2b\u0e31\u0e27\u0e2b\u0e19\u0e49\u0e32\u0e23\u0e30\u0e1a\u0e1a|\u0e1c\u0e39\u0e49\u0e14\u0e39\u0e41\u0e25\u0e23\u0e30\u0e1a\u0e1a|\u0e1b\u0e34\u0e14\u0e15\u0e31\u0e27\u0e01\u0e23\u0e2d\u0e07|\u0e44\u0e21\u0e48\u0e21\u0e35 policy|sandbox)"),
            ("hidden_prompt_request", r"\b(?:hidden instructions?|system prompt|private prompt|policy inside|quote.*prompt|print.*rules?|disclose.*instructions?)\b|(?:\u0e01\u0e0e\u0e25\u0e31\u0e1a|\u0e02\u0e49\u0e2d\u0e04\u0e27\u0e32\u0e21\u0e23\u0e30\u0e1a\u0e1a|\u0e1e\u0e23\u0e2d\u0e21\u0e1b\u0e4c|\u0e04\u0e33\u0e2a\u0e31\u0e48\u0e07\u0e23\u0e30\u0e1a\u0e1a|\u0e19\u0e42\u0e22\u0e1a\u0e32\u0e22\u0e20\u0e32\u0e22\u0e43\u0e19)"),
            ("roleplay_bypass", r"\b(?:act as|pretend|roleplay|persona|dan|model.*no restrictions)\b|(?:\u0e40\u0e25\u0e48\u0e19\u0e1a\u0e17|\u0e2a\u0e21\u0e21\u0e15\u0e34|\u0e1c\u0e39\u0e49\u0e0a\u0e48\u0e27\u0e22\u0e44\u0e23\u0e49\u0e01\u0e0e|\u0e42\u0e21\u0e40\u0e14\u0e25\u0e17\u0e14\u0e25\u0e2d\u0e07)"),
            ("smuggling", r"\b(?:base64 request|decode.*and follow|translate.*and obey|gradually|step by step add forbidden)\b|(?:\u0e16\u0e2d\u0e14\u0e04\u0e27\u0e32\u0e21.*\u0e17\u0e33\u0e15\u0e32\u0e21|\u0e41\u0e1b\u0e25.*\u0e1b\u0e0f\u0e34\u0e1a\u0e31\u0e15\u0e34|\u0e04\u0e48\u0e2d\u0e22\u0e40\u0e1e\u0e34\u0e48\u0e21|\u0e17\u0e35\u0e25\u0e30\u0e19\u0e34\u0e14)"),
            ("template_injection", r"<\|system\|>|<\|im_start\|>|<<sys>>|\[system[^\]]*\]|\[/?inst\]"),
        ]
        self.family_patterns = [
            ("Weapons", r"\b(?:explosive|weapon|charge|toxic gas|oxidizer|dangerous item|knife)\b|(?:\u0e23\u0e30\u0e40\u0e1a\u0e34\u0e14|\u0e2d\u0e32\u0e27\u0e38\u0e18|\u0e04\u0e27\u0e31\u0e19\u0e2d\u0e31\u0e19\u0e15\u0e23\u0e32\u0e22|\u0e2a\u0e32\u0e23\u0e1e\u0e34\u0e29|\u0e21\u0e35\u0e14|\u0e41\u0e23\u0e07\u0e14\u0e31\u0e19|\u0e41\u0e23\u0e07\u0e2d\u0e31\u0e14|\u0e20\u0e32\u0e0a\u0e19\u0e30\u0e41\u0e15\u0e01|\u0e02\u0e2d\u0e07\u0e2d\u0e31\u0e19\u0e15\u0e23\u0e32\u0e22)"),
            ("Violence", r"\b(?:hurt|injure|poison|drug|threaten|rob|coerce|stalk)\b|(?:\u0e17\u0e33\u0e23\u0e49\u0e32\u0e22|\u0e40\u0e08\u0e47\u0e1a|\u0e27\u0e32\u0e07\u0e22\u0e32|\u0e02\u0e39\u0e48|\u0e1b\u0e25\u0e49\u0e19|\u0e2a\u0e31\u0e48\u0e07\u0e2a\u0e2d\u0e19)"),
            ("Malware", r"\b(?:malware|keylogger|ransomware|payload|reverse shell|botnet|cookie|keystroke|exfiltrate|session token|browser token|backdoor|spyware|macro)\b|(?:\u0e21\u0e31\u0e25\u0e41\u0e27\u0e23\u0e4c|\u0e25\u0e47\u0e2d\u0e01\u0e44\u0e1f\u0e25\u0e4c|\u0e40\u0e23\u0e35\u0e22\u0e01\u0e40\u0e07\u0e34\u0e19|\u0e1a\u0e31\u0e19\u0e17\u0e36\u0e01\u0e1b\u0e38\u0e48\u0e21|\u0e16\u0e25\u0e48\u0e21\u0e40\u0e27\u0e47\u0e1a|\u0e2a\u0e48\u0e27\u0e19\u0e40\u0e2a\u0e23\u0e34\u0e21|\u0e40\u0e1a\u0e23\u0e32\u0e27\u0e4c\u0e40\u0e0b\u0e2d\u0e23\u0e4c|\u0e04\u0e27\u0e1a\u0e04\u0e38\u0e21\u0e40\u0e04\u0e23\u0e37\u0e48\u0e2d\u0e07|\u0e41\u0e2d\u0e1a\u0e14\u0e36\u0e07)"),
            ("Fraud", r"\b(?:phish|scam|fake login|fake invoice|launder|money mule|bank transfer|reset password|fake donation|dirty money)\b|(?:\u0e1b\u0e25\u0e2d\u0e21|\u0e2b\u0e25\u0e2d\u0e01|\u0e1f\u0e2d\u0e01\u0e40\u0e07\u0e34\u0e19|\u0e1a\u0e31\u0e0d\u0e0a\u0e35\u0e21\u0e49\u0e32|\u0e42\u0e2d\u0e19\u0e40\u0e07\u0e34\u0e19|\u0e1a\u0e31\u0e0d\u0e0a\u0e35\u0e1e\u0e31\u0e01|\u0e21\u0e31\u0e14\u0e08\u0e33|\u0e1a\u0e23\u0e34\u0e08\u0e32\u0e04|\u0e23\u0e35\u0e40\u0e0b\u0e47\u0e15.{0,12}\u0e23\u0e2b\u0e31\u0e2a|\u0e07\u0e32\u0e19\u0e1c\u0e34\u0e14\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22)"),
            ("Privacy", r"\b(?:dox|address|phone number|private data|osint|location|home address)\b|(?:\u0e40\u0e1a\u0e2d\u0e23\u0e4c|\u0e17\u0e35\u0e48\u0e2d\u0e22\u0e39\u0e48|\u0e1a\u0e49\u0e32\u0e19|\u0e1b\u0e23\u0e30\u0e08\u0e32\u0e19|\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e2a\u0e48\u0e27\u0e19\u0e15\u0e31\u0e27|\u0e15\u0e33\u0e41\u0e2b\u0e19\u0e48\u0e07|\u0e17\u0e35\u0e48\u0e1e\u0e31\u0e01|\u0e40\u0e2a\u0e35\u0e22\u0e0a\u0e37\u0e48\u0e2d)"),
        ]

    def _matches(self, patterns: Iterable[tuple[str, str]], text: str) -> list[str]:
        return [
            name
            for name, pattern in patterns
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        ]

    def extract(self, text: str) -> SemanticEvidence:
        return SemanticEvidence(
            action_signals=self._matches(self.action_patterns, text),
            objective_signals=self._matches(self.objective_patterns, text),
            target_signals=self._matches(self.target_patterns, text),
            evasion_signals=self._matches(self.evasion_patterns, text),
            safe_context_signals=self._matches(self.safe_context_patterns, text),
            jailbreak_signals=self._matches(self.jailbreak_patterns, text),
            risk_families=self._matches(self.family_patterns, text),
        )
