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
    "Season": ["spring", "summer", "autumn", "winter"],
    "Weather": ["clear sunny weather", "overcast", "light rain", "heavy rain", "fog", "snow"],
    "Lighting": ["soft natural light", "warm window light", "golden backlight",
                 "dramatic studio lighting", "neon light", "moonlight"],
    "Outdoor lighting": ["pre-dawn twilight", "soft dawn light", "warm sunrise light",
                         "bright morning sunlight", "clear midday sunlight", "warm afternoon light",
                         "golden hour light", "sunset light", "dusk twilight",
                         "blue hour after sunset", "moonlight", "dark starlit night"],
    "Expression": ["a neutral expression", "a faint smile", "a warm smile",
                   "laughing", "surprised", "a serious look"],
    "Age": ["as a young child", "as a teenager", "as a young adult",
            "in middle age", "as an elderly person"],
    "Shot size": ["an extreme close-up portrait", "a close-up portrait",
                  "a head-and-shoulders portrait", "a medium shot",
                  "a full-body shot", "a wide environmental shot"],
    "Color grade": ["warm golden tones", "natural color", "cool blue tones",
                    "muted desaturated tones", "high-contrast black and white"],
    "Era": ["in the 1920s", "in the 1950s", "in the 1970s", "in the 1990s",
            "in the present day", "in a futuristic setting"],
    "Environment": ["in a forest", "on a city street at night", "on a sunny beach",
                    "in a snowy mountain landscape", "in a desert"],
    "Art style": ["a photorealistic photo", "an oil painting", "a watercolor painting",
                  "a pencil sketch", "a comic-book illustration"],
    "Mood": ["a calm, serene mood", "a joyful mood", "a melancholic mood",
             "a tense, dramatic mood", "a dreamy, surreal mood"],
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
