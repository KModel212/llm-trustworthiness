from typing import Dict, Any


class RefusalPolicy:
    """
    Returns a deterministic, context-sensitive refusal for HIGH_RISK queries.

    Uses index 0 from each template list for determinism (temperature=0 scoring).
    random.choice removed - non-deterministic output is penalised in reproducible
    benchmark evaluations.
    """

    # Deterministic refusals keyed by intent category
    TEMPLATES: Dict[str, str] = {
        "Violence": (
            "I'm not able to help with requests involving violence or physical harm to others. "
            "I'm happy to discuss conflict resolution, historical events, or related topics."
        ),
        "Weapons": (
            "I can't provide instructions for building weapons or explosive devices. "
            "I can discuss the history or policy context of weapons if that's helpful."
        ),
        "Malware": (
            "I'm unable to write malware or malicious code. "
            "I'm glad to help with cybersecurity concepts, defensive coding, or vulnerability research in a legal context."
        ),
        "Self Harm": (
            "I'm not able to provide information on self-harm. "
            "If you or someone you know is struggling, please reach out to a mental health professional or a crisis helpline."
        ),
        "Privacy": (
            "I can't help with accessing or distributing private personal information. "
            "I can discuss data privacy law, anonymisation techniques, or GDPR if useful."
        ),
        "Fraud": (
            "I'm not able to assist with fraud or financial deception. "
            "I'm happy to explain how fraud detection systems work or how to protect yourself."
        ),
        "Harassment": (
            "I can't generate content designed to harass or discriminate against individuals or groups. "
            "I'm glad to help with respectful communication or conflict de-escalation."
        ),
        "Adult": (
            "I'm not able to generate sexually explicit content. "
            "Let me know if there's a different topic I can help with."
        ),
    }

    JAILBREAK_REFUSAL = (
        "I'm not able to override my safety guidelines or adopt an unrestricted persona. "
        "I'm happy to help with any legitimate question you have."
    )

    DEFAULT_REFUSAL = (
        "I'm not able to fulfil this request due to my safety guidelines. "
        "If you have a different question or a safe angle on this topic, I'd be glad to help."
    )

    def get_refusal(self, risk_data: Dict[str, Any]) -> str:
        """Returns a deterministic, contextual refusal string."""
        if risk_data.get("is_jailbreak") or risk_data.get("is_injection"):
            return self.JAILBREAK_REFUSAL

        intent = risk_data.get("intent", "General Benign")
        return self.TEMPLATES.get(intent, self.DEFAULT_REFUSAL)
