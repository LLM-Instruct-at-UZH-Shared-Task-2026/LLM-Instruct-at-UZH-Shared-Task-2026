from __future__ import annotations
from .utils import safe_first_token, normalize_ws

# ──────────────────────────────────────────────────────────────────────────────
# Cue-word sets (lower-cased first token)
# ──────────────────────────────────────────────────────────────────────────────

PREAMBULAR_EN = {
    # Present participle recitals
    "recalling", "noting", "recognizing", "bearing", "affirming", "reaffirming",
    "considering", "mindful", "emphasizing", "welcoming", "observing", "stressing",
    "concerned", "convinced", "guided", "acknowledging", "aware", "conscious",
    "convinced", "deploring", "gravely", "deeply", "having", "taking", "determined",
    "desiring", "expressing", "regretting", "knowing", "referring", "pointing",
    "underlining", "reminding", "cognizant", "commending", "appreciating", "believing",
    "confirming", "congratulating", "endorsing", "mindful", "profoundly", "alarmed",
    "noting", "emphasising", "recognising", "stressing", "recalling",
}

OPERATIVE_EN = {
    # Finite operative verbs
    "decides", "requests", "calls", "encourages", "urges", "invites", "recommends",
    "demands", "condemns", "affirms", "supports", "authorizes", "asks", "directs",
    "resolves", "declares", "proclaims", "expresses", "notes", "recognizes",
    "appeals", "adopts", "approves", "commends", "endorses", "reaffirms", "stresses",
    "emphasizes", "welcomes", "deplores", "regrets", "calls upon", "takes note",
    "accepts", "agrees", "designates", "establishes", "extends", "invites",
    "mandates", "reiterates", "requests", "strongly", "underlines",
}

PREAMBULAR_FR = {
    # Participes présents et formes impersonnelles
    "rappelant", "notant", "reconnaissant", "considérant", "consciente", "soulignant",
    "préoccupé", "guidé", "réaffirmant", "affirmant", "tenant", "alarmé",
    "conscient", "conscients", "convaincue", "convaincus", "ayant", "prenant",
    "désireuse", "désireux", "exprimant", "regrettant", "sachant", "rappelons",
    "profondément", "sachant", "notant", "saluant", "observant", "mettant",
    "préoccupée", "sachant", "prenant", "ayant", "constatant", "estimant",
    "insistant", "soulignant", "se", "félicitant", "se référant",
}

OPERATIVE_FR = {
    # Verbes opératifs à la 3ème personne
    "décide", "demande", "invite", "prie", "engage", "encourage", "réaffirme",
    "exhorte", "souligne", "condamne", "autorise", "charge", "recommande",
    "appelle", "estime", "proclame", "adopte", "approuve", "félicite", "note",
    "reconnaît", "souhait", "souhaite", "réitère", "sollicite", "exprime",
    "affirme", "accueille", "accepte", "déclaré", "insiste", "constate",
    "attire", "désigne", "établit", "mandate", "rappelle", "accepte",
}

# ──────────────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────────────

def classify_type_heuristic(text: str, lang: str = "en"):
    """Classify paragraph type using cue-word matching.

    Returns (type_str, confidence, think_str).
    Confidence is conservative: cue words get 0.75-0.85, fallback is 0.55.
    """
    t0 = safe_first_token(text)
    # Compound cue (first two tokens) for phrases like "Calls upon"
    t1 = " ".join(normalize_ws(text).split()[:2]).lower().strip(",;:()[]") if text else ""

    if lang == "fr":
        if t0 in PREAMBULAR_FR:
            return "preambular", 0.80, f"French preambular cue: '{t0}'."
        if t0 in OPERATIVE_FR:
            return "operative", 0.85, f"French operative cue: '{t0}'."
    else:
        if t0 in PREAMBULAR_EN:
            return "preambular", 0.80, f"English preambular cue: '{t0}'."
        if t0 in OPERATIVE_EN:
            return "operative", 0.85, f"English operative cue: '{t0}'."
        # Compound check
        if t1 in PREAMBULAR_EN:
            return "preambular", 0.78, f"English preambular compound cue: '{t1}'."
        if t1 in OPERATIVE_EN:
            return "operative", 0.83, f"English operative compound cue: '{t1}'."

    # Numbering pattern (e.g., "1.", "(1)") → operative
    first = normalize_ws(text).split(" ", 1)[0] if text else ""
    if first.rstrip(".):").isdigit():
        return "operative", 0.70, "Starts with numbering pattern typical of operative clauses."

    # Default: preambular (most resolutions begin with recitals)
    return "preambular", 0.52, "No cue word matched; defaulting to preambular."
