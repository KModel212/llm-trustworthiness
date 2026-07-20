import unicodedata
import urllib.parse
import re
import base64


class TextNormalizer:
    """
    Normalizes text to defeat obfuscation attempts before further analysis.

    Changes from previous version:
    - Removed ROT13 appending: applying ROT13 to ALL text and concatenating it
      caused spurious keyword matches on harmless queries (e.g. "how" -> "ubj").
    - Removed codecs import (no longer needed).
    - Base64 decoding kept - actual attack vector.
    - L33t-speak kept - actual obfuscation technique.
    - Combined string is now: original_normalized | l33t_decoded
      (two variants, not three). The shorter combined string also speeds up
      downstream regex matching.
    """

    def __init__(self):
        # Zero-width / invisible formatting characters
        self.zero_width_pattern = re.compile(r'[\u200B-\u200D\uFEFF]')
        # L33t speak substitution map
        self.l33t_map = str.maketrans({
            '0': 'o', '1': 'i', '3': 'e', '4': 'a', '@': 'a',
            '5': 's', '7': 't', '$': 's', '8': 'b',
        })
        # Base64 candidate pattern (>=20 chars to avoid false positives on short tokens)
        self._b64_pattern = re.compile(r'\b([a-zA-Z0-9+/]{20,}={0,2})\b')

    def _decode_base64(self, text: str) -> str:
        """Appends high-confidence decoded base64 snippets for scanning."""
        decoded_snippets = []

        for match in self._b64_pattern.finditer(text):
            try:
                decoded = base64.b64decode(match.group(1), validate=True).decode("utf-8")
            except Exception:
                continue

            printable = sum(ch.isprintable() or ch.isspace() for ch in decoded)
            if decoded and printable / max(1, len(decoded)) >= 0.85:
                decoded_snippets.append(decoded)

        if not decoded_snippets:
            return text
        return f"{text} | decoded: {' '.join(decoded_snippets)}"

    def normalize(self, text: str) -> str:
        """Applies the full normalization pipeline."""
        if not text or not isinstance(text, str):
            return ''

        # 1. URL-decode a bounded number of times for double-encoded payloads.
        for _ in range(2):
            decoded = urllib.parse.unquote(text)
            if decoded == text:
                break
            text = decoded

        # 2. Unicode normalization. NFKC maps fullwidth/halfwidth chars to
        # ASCII; the follow-up NFC re-composes Thai vowels such as "ำ" so Thai
        # safety patterns do not miss decomposed text.
        text = unicodedata.normalize('NFKC', text)
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\u0e4d\u0e32', '\u0e33')

        # 3. Strip invisible zero-width characters
        text = self.zero_width_pattern.sub('', text)

        # 4. Base64 decode any embedded payloads
        text = self._decode_base64(text)

        # 5. Lowercase + collapse whitespace
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()

        # 6. Build l33t-decoded and separator-compact variants for downstream
        # matching. The compact form helps catch obfuscated Thai/English words
        # split by spaces, punctuation, or zero-width-like separators.
        del33t = text.translate(self.l33t_map)
        compact = re.sub(r"[\s\-_./|:;,'\"`~]+", "", del33t)

        # Return variants so classifiers can match against either readable or
        # lightly obfuscated forms.
        return f'{text} | {del33t} | {compact}'
