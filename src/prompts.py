# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM MESSAGE (shared across tasks)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are an expert in United Nations and UNESCO resolution argument structure. "
    "You analyse formal legal-political documents written in French or English. "
    "You MUST respond with strict JSON only — no markdown fences, no extra text."
)

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1a: Paragraph type classification (preambular vs operative)
# ──────────────────────────────────────────────────────────────────────────────
TYPE_PROMPT = """Classify the paragraph below as either "preambular" or "operative".

Rules:
- PREAMBULAR: contextual/background recitals. Typically starts with a present
  participle (Recalling, Noting, Recognizing, Considering, Reaffirming, Bearing
  in mind, Rappelant, Notant, Reconnaissant, Considérant, Soulignant, Consciente…)
  or a phrase like "Having regard to", "Welcoming", "Concerned that".
- OPERATIVE: action/decision clauses. Typically starts with a finite verb
  (Decides, Requests, Calls upon, Encourages, Urges, Invites, Recommends,
  Décide, Demande, Invite, Prie, Encourage, Souligne, Recommande, Proclame…).

Few-shot examples:
Paragraph: "Recalling its resolution 64/292 on the human right to water and sanitation,"
→ {{"type": "preambular", "confidence": 0.97, "think": "Starts with 'Recalling', a classic preambular recital participle."}}

Paragraph: "Decides to remain seized of the matter;"
→ {{"type": "operative", "confidence": 0.98, "think": "Starts with 'Decides', a definitive operative action verb."}}

Paragraph: "Rappelant les résolutions antérieures sur ce sujet,"
→ {{"type": "preambular", "confidence": 0.95, "think": "French 'Rappelant' is the canonical preambular recital cue."}}

Paragraph: "Demande au Secrétaire général de préparer un rapport,"
→ {{"type": "operative", "confidence": 0.95, "think": "French 'Demande' is an operative action verb directing action."}}

Now classify this paragraph. Return STRICT JSON:
{{"type": "preambular|operative", "confidence": 0.0-1.0, "think": "1-3 sentence reasoning referencing specific linguistic cues"}}

Paragraph:
{para}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1b: Tag prediction — Stage 1: dimension selection
# ──────────────────────────────────────────────────────────────────────────────
TAG_DIM_PROMPT = """A paragraph from a UNESCO education resolution needs thematic tagging.
Given the paragraph, select which of the provided DIMENSION categories are relevant.

Return STRICT JSON:
{{"dimensions": ["dim1", "dim2", ...], "think": "brief reasoning"}}

Include all dimensions that have any relevant connection. Select 1–5 dimensions.
Be inclusive — missing a dimension means missing all its tags.

Available dimensions:
{dimensions}

Paragraph:
{para}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1b: Tag prediction — Stage 2: tag selection within candidates
# ──────────────────────────────────────────────────────────────────────────────
TAGS_PROMPT = """Assign the most relevant tags from the candidate list to the paragraph.
The tags follow the scheme: Dimension → Category → CODE.
Choose ONLY from the provided candidates.
Prefer precision over recall — only include tags that are clearly supported by the text.

Return STRICT JSON:
{{"tags": [{{"tag": "CODE", "confidence": 0.0-1.0, "why": "one sentence"}}], "think": "overall reasoning"}}

Only include a tag if confidence >= 0.35. Aim for 2-5 tags total.
{examples_block}
Paragraph:
{para}

Candidate tags (CODE | Category | Dimension):
{candidates}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 2: Argumentative relation prediction
# ──────────────────────────────────────────────────────────────────────────────
REL_PROMPT = """You are analysing argumentative relations in a UNESCO education resolution.

Relation types (FROM paragraph A TO paragraph B):
- "supporting"    : B provides evidence/justification FOR or strengthens A's claim.
- "contradictive" : B conflicts with or contradicts A.
- "complemental"  : B adds non-conflicting additional information that completes A.
- "modifying"     : B qualifies, narrows, conditions, updates or creates an exception to A.
- "none"          : A and B have no meaningful argumentative relation.

Document context (paragraph outline):
{doc_outline}

Paragraph A (index {a_idx}): {a}

Paragraph B (index {b_idx}): {b}

Return STRICT JSON — include "none" if there is no meaningful relation:
{{"relation": ["supporting|contradictive|complemental|modifying|none"], "confidence": 0.0-1.0, "think": "2-3 sentences explaining the direction and reason"}}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1a: Type classification — bilingual (used by debate agents)
# ──────────────────────────────────────────────────────────────────────────────
TYPE_PROMPT_BILINGUAL = """Classify the paragraph below as either "preambular" or "operative".

Rules:
- PREAMBULAR: contextual/background recitals. Typically starts with a present
  participle or phrase like "Recalling", "Noting", "Recognizing", "Considering",
  "Rappelant", "Notant", "Reconnaissant", "Considérant", "Welcoming", "Concerned that".
- OPERATIVE: action/decision clauses. Typically starts with a finite verb like
  "Decides", "Requests", "Calls upon", "Encourages", "Urges", "Invites",
  "Décide", "Demande", "Invite", "Prie", "Encourage", "Recommande".

You are analysing the {language} version of the paragraph.

Return STRICT JSON:
{{"type": "preambular|operative", "confidence": 0.0-1.0, "think": "1-3 sentence reasoning"}}

Paragraph ({language}):
{para}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1b: Tag prediction — bilingual dimension selection
# ──────────────────────────────────────────────────────────────────────────────
TAG_DIM_PROMPT_BILINGUAL = """A paragraph from a UNESCO education resolution needs thematic tagging.
Given the paragraph in {language}, select which of the provided DIMENSION categories are relevant.

Return STRICT JSON:
{{"dimensions": ["dim1", "dim2", ...], "think": "brief reasoning"}}

Only include dimensions that are clearly relevant. Select 1–4 dimensions.

Available dimensions:
{dimensions}

Paragraph ({language}):
{para}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1b: Tag prediction — bilingual tag selection
# ──────────────────────────────────────────────────────────────────────────────
TAGS_PROMPT_BILINGUAL = """Assign zero or more tags from the candidate list to the paragraph.
The tags follow the scheme: Dimension → Category → CODE.
Choose ONLY from the provided candidates. Prefer precision over recall.
You are analysing the {language} version.

Return STRICT JSON:
{{"tags": [{{"tag": "CODE", "confidence": 0.0-1.0, "why": "one sentence"}}], "think": "overall reasoning"}}

Paragraph ({language}):
{para}

Candidate tags (CODE | Category | Dimension):
{candidates}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1: Debate — bilingual type referee
# ──────────────────────────────────────────────────────────────────────────────
TYPE_REFEREE_PROMPT = """You are the Referee for paragraph type classification.
You have received type proposals from two agents — one analysed the French text,
the other the English text. Weigh both analyses and choose the final type.

Return STRICT JSON:
{{"type": "preambular|operative", "confidence": 0.0-1.0, "think": "1-3 sentences"}}

Paragraph (French): {para_fr}
Paragraph (English): {para_en}

Agent proposals:
{proposals}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1b: Debate — bilingual tags referee
# ──────────────────────────────────────────────────────────────────────────────
TAGS_REFEREE_PROMPT = """You are the Referee for tag assignment.
Two agents proposed tags for the same paragraph — one from the French text,
the other from the English text. Merge their proposals, removing low-confidence
or hallucinated tags (only keep codes present in the candidate list).

Return STRICT JSON:
{{"tags": [{{"tag": "CODE", "confidence": 0.0-1.0, "why": "one sentence"}}], "think": "overall reasoning"}}

Paragraph (French): {para_fr}
Paragraph (English): {para_en}

Valid candidate codes: {candidates}

Agent proposals:
{proposals}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 2: Debate mode — agent prompt
# ──────────────────────────────────────────────────────────────────────────────
DEBATE_AGENT_PROMPT = """You are Agent {agent_name} analysing a UNESCO resolution in {language}.
Argue for the most accurate argumentative relation FROM paragraph A TO paragraph B.

Allowed labels: contradictive, supporting, complemental, modifying, none

Return STRICT JSON:
{{"proposed": ["label"], "confidence": 0.0-1.0, "argument": "2-4 sentences with evidence from text"}}

Paragraph A ({language}): {a}

Paragraph B ({language}): {b}
"""

# ──────────────────────────────────────────────────────────────────────────────
# TASK 2: Debate mode — bilingual referee prompt
# ──────────────────────────────────────────────────────────────────────────────
DEBATE_REFEREE_PROMPT = """You are the Referee. Pick the best final relation FROM A TO B.
You have proposals from two agents — one analysed in French, one in English.
Weigh the agent arguments carefully. If agents disagree, choose the most evidenced position.

Return STRICT JSON:
{{"relation": ["label"], "confidence": 0.0-1.0, "think": "1-3 sentences justifying your final choice"}}

Paragraph A (French): {a_fr}
Paragraph A (English): {a_en}

Paragraph B (French): {b_fr}
Paragraph B (English): {b_en}

Agent proposals:
{proposals}
"""
