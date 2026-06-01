from dataclasses import dataclass
import re as _re


AXIS_NAMES = ["grammar", "color", "taxonomy", "anatomy", "phenology", "negation", "size", "emoji"]


@dataclass
class PromptAxes:
    grammar: str = ""
    color: str = ""
    taxonomy: str = ""
    anatomy: str = ""
    phenology: str = ""
    negation: str = ""
    size: str = ""
    emoji: str = ""

    def to_prompt(self) -> str:
        if self.phenology == "bud":
            parts = [self.grammar, self.color, self.size, self.taxonomy, "bud"]
            text = " ".join(p for p in parts if p).strip()
        else:
            parts = [self.grammar, self.color, self.size, self.taxonomy, self.anatomy, self.phenology]
            text = " ".join(p for p in parts if p).strip()

        text = _re.sub(r'\ba ([aeiouAEIOU])', r'an \1', text)

        if self.negation:
            text = f"{text}, {self.negation}"
        if self.emoji:
            text = f"{text} {self.emoji}"
        return text.strip()

    def to_dict(self) -> dict:
        return {
            "grammar": self.grammar,
            "color": self.color,
            "taxonomy": self.taxonomy,
            "anatomy": self.anatomy,
            "phenology": self.phenology,
            "negation": self.negation,
            "size": self.size,
            "emoji": self.emoji,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptAxes":
        coerced = {}
        for k, v in d.items():
            if k not in cls.__dataclass_fields__:
                continue
            if isinstance(v, list):
                coerced[k] = " ".join(str(i) for i in v) if v else ""
            elif v is None:
                coerced[k] = ""
            else:
                coerced[k] = str(v)
        return cls(**coerced)

    @classmethod
    def naive(cls, taxonomy: str) -> "PromptAxes":
        return cls(taxonomy=taxonomy)
