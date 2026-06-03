"""Input guardrails and lightweight language routing for text queries."""

from __future__ import annotations

import re
import unicodedata
from typing import Final


VIETNAMESE_DIACRITIC_RE: Final[re.Pattern[str]] = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
    r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)
ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-zÀ-ỹ0-9]")

VIETNAMESE_HINTS: Final[frozenset[str]] = frozenset(
    {
        "ao",
        "áo",
        "ba lo",
        "balo",
        "ba lô",
        "co",
        "cô",
        "chiec",
        "chiếc",
        "chu",
        "chú",
        "den",
        "đen",
        "dep",
        "dép",
        "do",
        "đỏ",
        "doi",
        "đội",
        "giay",
        "giày",
        "mac",
        "mặc",
        "mu",
        "mũ",
        "nam",
        "non",
        "nón",
        "nguoi",
        "người",
        "nu",
        "nữ",
        "quan",
        "quần",
        "tim",
        "tím",
        "tim",
        "tìm",
        "tui",
        "túi",
        "vang",
        "vàng",
        "xanh",
        "xam",
        "xám",
    }
)

ENGLISH_HINTS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "backpack",
        "bag",
        "black",
        "blue",
        "carrying",
        "dress",
        "female",
        "green",
        "hat",
        "in",
        "jacket",
        "male",
        "man",
        "pants",
        "person",
        "purse",
        "red",
        "shirt",
        "shoes",
        "shorts",
        "sneakers",
        "the",
        "wearing",
        "white",
        "woman",
        "yellow",
    }
)


class QueryValidator:
    """Validate raw text and infer whether it is primarily English or Vietnamese."""

    MIN_QUERY_LENGTH: Final[int] = 5

    def validate_and_detect(self, query: str) -> dict[str, bool | str]:
        """Clean, reject garbage inputs, and route to ``en`` or ``vi``.

        Args:
            query: Raw user text describing a pedestrian.

        Returns:
            A dictionary with ``is_valid``, ``language``, and ``error_code``.
        """

        cleaned = (query or "").strip()
        compact = re.sub(r"\s+", "", cleaned)

        if len(cleaned) < self.MIN_QUERY_LENGTH:
            return self._invalid("QUERY_TOO_SHORT")
        if compact.isdecimal():
            return self._invalid("QUERY_ONLY_NUMBERS")
        if not ALNUM_RE.search(cleaned):
            return self._invalid("QUERY_ONLY_SPECIAL_CHARS")

        return {
            "is_valid": True,
            "language": self._detect_language(cleaned),
            "error_code": "",
        }

    @staticmethod
    def _invalid(error_code: str) -> dict[str, bool | str]:
        """Return a standardized invalid validation result."""

        return {
            "is_valid": False,
            "language": "unknown",
            "error_code": error_code,
        }

    def _detect_language(self, text: str) -> str:
        """Detect VI/EN using diacritics plus small domain dictionaries."""

        normalized = text.lower()
        ascii_text = self._strip_accents(normalized)
        tokens = set(re.findall(r"[a-zA-ZÀ-ỹ]+", normalized))
        ascii_tokens = set(re.findall(r"[a-zA-Z]+", ascii_text))

        vi_score = 2 if VIETNAMESE_DIACRITIC_RE.search(text) else 0
        vi_score += sum(1 for hint in VIETNAMESE_HINTS if hint in tokens)
        vi_score += sum(1 for hint in VIETNAMESE_HINTS if hint in ascii_tokens)
        vi_score += sum(1 for hint in VIETNAMESE_HINTS if " " in hint and hint in normalized)

        en_score = sum(1 for hint in ENGLISH_HINTS if hint in ascii_tokens)

        return "vi" if vi_score > en_score else "en"

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Remove combining marks while keeping Vietnamese ``đ`` distinct."""

        text = text.replace("đ", "d").replace("Đ", "D")
        normalized = unicodedata.normalize("NFD", text)
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn")
