"""Prompt-travel template dimensions for LoRA Royale.

Each dimension is an ordered list of words that reads as a smooth continuum
(dawn -> night, spring -> winter, child -> elderly). Prompt travel encodes the
base prompt at each waypoint and interpolates the *text embedding* between them
on a fixed seed, so the same subject morphs through the dimension.

The base prompt may contain a `{x}` slot where the waypoint word goes; if it has
no slot, the word is appended (", <word>").
"""

from typing import List

SLOT = "{x}"

# Ordered, continuum-friendly dimensions. Order matters — travel walks the list
# start -> end, so each neighbour should be a small conceptual step.
TEMPLATES = {
    "Time of day": ["dawn", "early morning", "midday", "golden hour", "dusk", "night"],
    "Season": ["early spring", "late spring", "early summer", "high summer",
               "early autumn", "late autumn", "early winter", "deep winter"],
    "Weather": ["clear sunny weather", "a few scattered clouds", "partly cloudy", "overcast",
                "light drizzle", "steady rain", "heavy rain", "a thunderstorm",
                "misty fog", "light snowfall", "heavy snow"],
    "Lighting": ["soft natural light", "warm window light", "golden backlight",
                 "dramatic studio lighting", "neon light", "moonlight"],
    "Outdoor lighting": ["pre-dawn twilight", "soft dawn light", "warm sunrise light",
                         "bright morning sunlight", "clear midday sunlight", "warm afternoon light",
                         "golden hour light", "sunset light", "dusk twilight",
                         "blue hour after sunset", "moonlight", "dark starlit night"],
    "Expression": ["a serious expression", "a neutral expression", "a subtle smile",
                   "a soft smile", "a gentle smile", "a warm smile", "a cheerful smile",
                   "a wide grin", "a joyful grin", "laughing", "laughing hard",
                   "a surprised expression"],
    "Age": ["as a baby", "as a toddler", "as a young child", "as a pre-teen",
            "as a teenager", "as a young adult", "in their late twenties",
            "in their thirties", "in their forties", "in middle age",
            "in their late fifties", "as a senior", "as an elderly person",
            "as a very old person"],
    "Shot size": ["an extreme close-up", "a close-up portrait", "a head-and-shoulders portrait",
                  "a waist-up shot", "a medium shot", "a three-quarter shot",
                  "a full-body shot", "a wide shot", "a wide environmental shot"],
    "Color grade": ["warm golden tones", "warm natural tones", "natural color",
                    "cool tones", "cool blue tones", "teal-and-orange grade",
                    "muted desaturated tones", "high-contrast", "high-contrast black and white"],
    "Era": ["in the 1900s", "in the 1920s", "in the 1940s", "in the 1950s", "in the 1960s",
            "in the 1970s", "in the 1980s", "in the 1990s", "in the 2000s",
            "in the present day", "in a near-future setting", "in a far-future setting"],
    "Environment": ["in a dense forest", "in a sunlit meadow", "by a calm lake",
                    "on a sandy beach", "on a busy city street", "in a quiet town square",
                    "in snowy mountains", "in a desert", "on a misty moor"],
    "Art style": ["a photorealistic photo", "a hyperrealistic painting", "an oil painting",
                  "an impressionist painting", "a watercolor painting", "an ink drawing",
                  "a pencil sketch", "a comic-book illustration", "a cel-shaded anime style",
                  "a low-poly 3D render"],
    "Mood": ["a calm, serene mood", "a peaceful mood", "a contemplative mood",
             "a joyful mood", "an energetic mood", "a melancholic mood", "a somber mood",
             "a tense, dramatic mood", "a mysterious mood", "a dreamy, surreal mood"],
}

# Suggested dimensions list for a dropdown (templates + Custom appended by the GUI).
DIMENSION_NAMES = list(TEMPLATES.keys())


def waypoint_prompt(base: str, word: str) -> str:
    """Fill the base prompt's `{x}` slot with `word`, or append it if no slot."""
    base = base or ""
    if SLOT in base:
        return base.replace(SLOT, word)
    if not base.strip():
        return word
    return f"{base.rstrip().rstrip(',')}, {word}"


def build_waypoint_prompts(base: str, words: List[str]) -> List[str]:
    return [waypoint_prompt(base, w) for w in words]


def parse_custom(text: str) -> List[str]:
    """Comma-separated custom waypoints -> cleaned list."""
    return [w.strip() for w in (text or "").split(",") if w.strip()]


def dominant_word(words: List[str], t: float) -> str:
    """The waypoint word nearest to fraction `t` in [0,1] — used for the burned-in
    badge so the clip ticks through the dimension as it morphs."""
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    pos = min(max(float(t), 0.0), 1.0) * (len(words) - 1)
    return words[int(round(pos))]
