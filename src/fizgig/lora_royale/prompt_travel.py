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
    "Time of day detailed": [
        "at dawn, soft pale early light, cool blue tones, gentle long shadows, a calm quiet sky",
        "in early morning, soft warm low sunlight, fresh clear light, long soft shadows",
        "at midday, bright harsh overhead sun, strong direct light, short hard shadows",
        "at golden hour, warm low-angle sun, a golden glow, long soft shadows, warm rim light",
        "at dusk, fading warm light, a deep orange and purple sky, soft dim shadows",
        "at night, a dark sky, dim ambient light, deep shadows, cool moonlit tones",
    ],
    "Season": ["early spring", "late spring", "early summer", "high summer",
               "early autumn", "late autumn", "early winter", "deep winter"],
    "Season detailed": [
        "in early spring, fresh green buds, budding trees, soft cool light, scattered early blooms",
        "in late spring, lush green foliage, blossoming flowers, bright fresh light",
        "in early summer, vibrant green leaves, warm bright sunlight, a clear blue sky",
        "in high summer, lush dense greenery, hot bright sun, deep warm light",
        "in early autumn, leaves turning gold and amber, warm mellow light, crisp air",
        "in late autumn, fallen orange and brown leaves, bare branches, soft grey light",
        "in early winter, frosty bare trees, cold pale light, the first light snow",
        "in deep winter, heavy snow, bare frozen trees, cold blue-grey light, icy air",
    ],
    "Weather": ["clear sunny weather", "a few scattered clouds", "partly cloudy", "overcast",
                "light drizzle", "steady rain", "heavy rain", "a thunderstorm",
                "misty fog", "light snowfall", "heavy snow"],
    "Weather detailed": [
        "in clear sunny weather, bright direct sunlight, a blue sky, crisp hard shadows",
        "with a few scattered clouds, bright sun, mostly blue sky, soft drifting shadows",
        "in partly cloudy weather, patchy sun and cloud, soft diffused light",
        "in overcast weather, a flat grey cloudy sky, soft even diffused light, no harsh shadows",
        "in light drizzle, fine misty rain, damp surfaces, soft grey light",
        "in steady rain, falling raindrops, wet glistening surfaces, dull grey light",
        "in heavy rain, a heavy downpour, soaked dripping surfaces, dark stormy light",
        "in a thunderstorm, dark dramatic storm clouds, lightning, heavy rain, moody light",
        "in misty fog, thick haze, low visibility, soft diffuse muted light, a pale background",
        "in light snowfall, gentle falling snowflakes, a light dusting of snow, soft cold light",
        "in heavy snow, thick falling snow, deep white snow cover, cold flat bright light",
    ],
    "Lighting": ["soft natural light", "warm window light", "golden backlight",
                 "dramatic studio lighting", "neon light", "moonlight"],
    "Lighting detailed": [
        "in soft natural light, gentle even diffused illumination, smooth soft shadows",
        "in warm window light, a soft directional glow from a window, gentle warm tones",
        "in golden backlight, warm sun behind the subject, a glowing rim light, soft lens flare",
        "in dramatic studio lighting, a strong directional key light, deep contrast, dark shadows",
        "in neon light, colorful glowing signs, saturated pink and blue glow, moody reflections",
        "in moonlight, a cool blue nighttime glow, soft dim light, deep shadows",
    ],
    "Outdoor lighting": ["pre-dawn twilight", "soft dawn light", "warm sunrise light",
                         "bright morning sunlight", "clear midday sunlight", "warm afternoon light",
                         "golden hour light", "sunset light", "dusk twilight",
                         "blue hour after sunset", "moonlight", "dark starlit night"],
    "Outdoor lighting detailed": [
        "in pre-dawn twilight, dim cool blue light before sunrise, a soft dark sky, a faint glow on the horizon",
        "in soft dawn light, pale cool early sunlight, gentle long shadows, a calm sky",
        "in warm sunrise light, a low golden sun on the horizon, a warm orange glow, long shadows",
        "in bright morning sunlight, fresh clear light, a warm low sun, crisp shadows",
        "in clear midday sunlight, a strong overhead sun, bright harsh light, short hard shadows",
        "in warm afternoon light, soft warm sun, gentle lengthening shadows",
        "in golden hour light, a warm low-angle sun, a golden glow, long soft shadows, warm rim light",
        "in sunset light, a glowing orange and pink sky, a warm fading sun, long shadows",
        "at dusk, fading light, a deep blue and purple sky, soft dim shadows",
        "in the blue hour after sunset, deep cool blue light, a soft glow, dim ambient light",
        "in moonlight, a cool blue nighttime glow, soft dim light, deep shadows",
        "on a dark starlit night, a deep dark sky, faint cool starlight, very low ambient light",
    ],
    "Expression": ["a serious expression", "a neutral expression", "a subtle smile",
                   "a soft smile", "a gentle smile", "a warm smile", "a cheerful smile",
                   "a wide grin", "a joyful grin", "laughing", "laughing hard",
                   "a surprised expression"],
    "Expression detailed": [
        "with a serious expression, level brows, a firm set mouth, a steady gaze",
        "with a neutral expression, a relaxed face, calm eyes, a softly closed mouth",
        "with a subtle smile, faintly upturned lips, soft relaxed eyes",
        "with a soft smile, gently curved lips, warm relaxed eyes",
        "with a gentle smile, lightly raised cheeks, soft kind eyes",
        "with a warm smile, raised cheeks, lightly crinkled eyes, slightly parted lips",
        "with a cheerful smile, bright eyes, raised cheeks, a clear happy smile",
        "with a wide grin, a broad smile showing teeth, lifted cheeks, bright eyes",
        "with a joyful grin, a big open smile, scrunched happy eyes, full cheeks",
        "laughing, an open mouth mid-laugh, scrunched eyes, raised cheeks, a joyful face",
        "laughing hard, a wide open mouth, eyes squeezed shut, head tilted back, full joy",
        "with a surprised expression, wide open eyes, raised brows, an open mouth",
    ],
    "Age": ["as a baby", "as a toddler", "as a young child", "as a pre-teen",
            "as a teenager", "as a young adult", "in their late twenties",
            "in their thirties", "in their forties", "in middle age",
            "in their late fifties", "as a senior", "as an elderly person",
            "as a very old person"],
    "Age detailed": [
        "as a baby with soft smooth skin, chubby round cheeks, wispy fine hair, and big round eyes",
        "as a toddler with soft baby skin, round full cheeks, fine soft hair, and a tiny young face",
        "as a young child with smooth unblemished skin, bright eyes, full cheeks, and a small youthful face",
        "as a pre-teen with a youthful unlined face, smooth clear skin, and bright eyes",
        "as a teenager with a fresh youthful face, smooth clear skin, and a slim young build",
        "as a young adult in their early twenties with smooth clear skin, a fresh face, and bright eyes",
        "in their late twenties with clear healthy skin, a mature youthful face, and a defined jaw",
        "in their thirties with a mature face, the first faint fine lines, and slightly less youthful skin",
        "in their forties with visible fine lines, light wrinkles around the eyes, and a mature lined face",
        "in middle age with clear wrinkles, forehead lines, slight greying hair, and aging skin",
        "in their late fifties with greying hair, deeper wrinkles, loosening skin, and an aging face",
        "as a senior with grey hair, deep wrinkles, age spots, and weathered sagging skin",
        "as an elderly person with white hair, deep heavy wrinkles, sagging loose skin, and a frail aged face",
        "as a very old person with thin white hair, deeply creased sagging skin, sunken features, and a frail elderly face",
    ],
    "Shot size": ["an extreme close-up", "a close-up portrait", "a head-and-shoulders portrait",
                  "a waist-up shot", "a medium shot", "a three-quarter shot",
                  "a full-body shot", "a wide shot", "a wide environmental shot"],
    "Shot size detailed": [
        "an extreme close-up, tightly framed on the face, filling the frame with fine detail",
        "a close-up portrait, the face filling the frame, shallow depth of field",
        "a head-and-shoulders portrait, framed from the chest up, a classic portrait crop",
        "a waist-up shot, framed from the waist up, showing the upper body",
        "a medium shot, framed from roughly the knees up, balanced framing",
        "a three-quarter shot, framed from mid-thigh up, most of the body visible",
        "a full-body shot, the entire figure visible head to toe in frame",
        "a wide shot, the full figure small in frame with surrounding space",
        "a wide environmental shot, the small figure within a large surrounding scene",
    ],
    "Color grade": ["warm golden tones", "warm natural tones", "natural color",
                    "cool tones", "cool blue tones", "teal-and-orange grade",
                    "muted desaturated tones", "high-contrast", "high-contrast black and white"],
    "Color grade detailed": [
        "graded in warm golden tones, rich amber highlights, warm cozy color",
        "graded in warm natural tones, soft warm color, natural skin tones",
        "in natural color, balanced neutral tones, true-to-life color",
        "graded in cool tones, soft blue-leaning color, a calm cool palette",
        "graded in cool blue tones, blue shadows, a cold crisp palette",
        "in a teal-and-orange grade, orange skin tones against teal shadows, cinematic color",
        "in muted desaturated tones, low saturation, soft faded color",
        "in a high-contrast grade, deep blacks, bright highlights, punchy color",
        "in high-contrast black and white, deep blacks, bright whites, dramatic monochrome",
    ],
    "Era": ["in the 1900s", "in the 1920s", "in the 1940s", "in the 1950s", "in the 1960s",
            "in the 1970s", "in the 1980s", "in the 1990s", "in the 2000s",
            "in the present day", "in a near-future setting", "in a far-future setting"],
    "Era detailed": [
        "in the 1900s, early-1900s setting, vintage Edwardian styling, a sepia-toned antique photo look",
        "in the 1920s, roaring-twenties styling, vintage 1920s fashion, old film grain",
        "in the 1940s, 1940s wartime-era styling, vintage 1940s fashion, a classic monochrome film look",
        "in the 1950s, mid-century 1950s styling, retro 1950s fashion, faded vintage color",
        "in the 1960s, mod 1960s styling, retro 1960s fashion, warm vintage film color",
        "in the 1970s, retro 1970s styling, seventies fashion, warm faded film tones",
        "in the 1980s, bold 1980s styling, eighties fashion, saturated retro color",
        "in the 1990s, nineties styling, 1990s fashion, a slightly grainy film look",
        "in the 2000s, early-2000s styling, Y2K fashion, an early digital camera look",
        "in the present day, modern contemporary styling, current fashion, a clean digital photo",
        "in a near-future setting, sleek modern futuristic styling, subtle sci-fi details",
        "in a far-future setting, advanced futuristic styling, sci-fi technology and design",
    ],
    "Environment": ["in a dense forest", "in a sunlit meadow", "by a calm lake",
                    "on a sandy beach", "on a busy city street", "in a quiet town square",
                    "in snowy mountains", "in a desert", "on a misty moor"],
    "Environment detailed": [
        "in a dense forest, tall trees, green foliage, dappled light through the leaves",
        "in a sunlit meadow, an open grassy field, wildflowers, a bright open sky",
        "by a calm lake, still reflective water, a distant shoreline, an open sky",
        "on a sandy beach, golden sand, ocean waves, a bright open sky",
        "on a busy city street, tall buildings, traffic and crowds, an urban backdrop",
        "in a quiet town square, old buildings, cobblestones, calm open space",
        "in snowy mountains, snow-capped peaks, cold crisp air, a vast landscape",
        "in a desert, golden sand dunes, dry barren land, a harsh bright sun",
        "on a misty moor, rolling hills, low fog, a muted grey-green landscape",
    ],
    "Art style": ["a photorealistic photo", "a hyperrealistic painting", "an oil painting",
                  "an impressionist painting", "a watercolor painting", "an ink drawing",
                  "a pencil sketch", "a comic-book illustration", "a cel-shaded anime style",
                  "a low-poly 3D render"],
    "Art style detailed": [
        "as a photorealistic photo, sharp realistic detail, lifelike skin and lighting",
        "as a hyperrealistic painting, painted yet lifelike detail, smooth realistic rendering",
        "as an oil painting, visible brushstrokes, rich textured paint, a classical painterly style",
        "as an impressionist painting, loose visible brushstrokes, soft blended color, dappled light",
        "as a watercolor painting, soft translucent washes, bleeding colors, paper texture",
        "as an ink drawing, bold black ink lines, crosshatching, high-contrast linework",
        "as a pencil sketch, graphite shading, soft sketch lines, hand-drawn paper texture",
        "as a comic-book illustration, bold outlines, flat inked colors, halftone shading",
        "in a cel-shaded anime style, clean line art, flat cel shading, vibrant anime colors",
        "as a low-poly 3D render, faceted geometric shapes, flat shaded polygons, a simple CG look",
    ],
    "Mood": ["a calm, serene mood", "a peaceful mood", "a contemplative mood",
             "a joyful mood", "an energetic mood", "a melancholic mood", "a somber mood",
             "a tense, dramatic mood", "a mysterious mood", "a dreamy, surreal mood"],
    "Mood detailed": [
        "with a calm, serene mood, a peaceful tranquil atmosphere, soft gentle tones",
        "with a peaceful mood, a relaxed quiet atmosphere, soft soothing light",
        "with a contemplative mood, a thoughtful quiet atmosphere, soft pensive light",
        "with a joyful mood, a bright happy atmosphere, warm uplifting light",
        "with an energetic mood, a vibrant dynamic atmosphere, bright punchy light",
        "with a melancholic mood, a wistful sombre atmosphere, soft muted cool light",
        "with a somber mood, a heavy serious atmosphere, dim subdued light",
        "with a tense, dramatic mood, a high-contrast atmosphere, deep shadows, moody light",
        "with a mysterious mood, a shadowy enigmatic atmosphere, dim moody light",
        "with a dreamy, surreal mood, a soft hazy ethereal atmosphere, glowing diffuse light",
    ],
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
