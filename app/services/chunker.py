import re

from app.logger import get_logger

logger = get_logger(__name__)

MAX_CHUNK_SIZE = 1500
OVERLAP_SIZE = 200

# Mapping of regex keyword groups to normalized section types
SECTION_KEYWORDS: dict[str, list[str]] = {
    "summary": [
        "summary", "objective", "profile", "about", "about me",
        "professional summary", "career objective", "personal statement",
        "executive summary", "career summary",
    ],
    "experience": [
        "experience", "work experience", "employment", "professional experience",
        "work history", "employment history", "career history",
        "relevant experience", "professional background",
    ],
    "education": [
        "education", "academic", "academic background", "qualifications",
        "educational background", "academic qualifications",
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "key skills", "areas of expertise", "expertise", "technologies",
        "technical competencies", "proficiencies", "tools",
    ],
    "projects": [
        "projects", "personal projects", "key projects", "selected projects",
        "notable projects", "academic projects",
    ],
    "certifications": [
        "certifications", "certificates", "licenses", "credentials",
        "professional certifications", "licenses and certifications",
        "certifications and licenses",
    ],
    "awards": [
        "awards", "achievements", "honors", "accomplishments",
        "awards and honors", "recognition",
    ],
    "languages": [
        "languages", "language skills", "language proficiency",
    ],
    "references": [
        "references", "professional references",
    ],
}

# Build keyword list sorted longest-first so longer phrases match before shorter ones.
_all_keywords = sorted(
    [(kw, stype) for stype, keywords in SECTION_KEYWORDS.items() for kw in keywords],
    key=lambda x: len(x[0]),
    reverse=True,
)

_keyword_pattern = "|".join(re.escape(kw) for kw, _ in _all_keywords)

SECTION_HEADER_RE = re.compile(
    rf"^\s*({_keyword_pattern})\s*[:—\-]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_KEYWORD_TO_TYPE: dict[str, str] = {
    kw.lower(): stype for kw, stype in _all_keywords
}

_SENTENCE_BREAK_RE = re.compile(r"[.!?]\s", re.DOTALL)


def _normalize_section_type(header_text: str) -> str:
    """Map a detected header string to a normalized section type."""
    cleaned = header_text.strip().rstrip(":").rstrip("-").rstrip("\u2014").strip().lower()
    if cleaned in _KEYWORD_TO_TYPE:
        return _KEYWORD_TO_TYPE[cleaned]
    # Fallback: partial match against keywords
    for kw, stype in _all_keywords:
        if kw in cleaned or cleaned in kw:
            return stype
    return "other"


def detect_sections(text: str) -> list[tuple[str, str]]:
    """Detect resume sections by scanning for header lines.

    Returns a list of (section_type, section_text) tuples in document order.
    If no sections are detected, returns a single ("full_resume", text) tuple.
    Text before the first detected header is returned as ("header", ...).
    """
    matches = list(SECTION_HEADER_RE.finditer(text))

    if not matches:
        logger.debug("No section headers detected; treating entire text as full_resume")
        return [("full_resume", text.strip())]

    sections: list[tuple[str, str]] = []

    first_start = matches[0].start()
    header_text = text[:first_start].strip()
    if header_text:
        sections.append(("header", header_text))

    for i, match in enumerate(matches):
        section_type = _normalize_section_type(match.group(1))
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[content_start:content_end].strip()

        if section_text:
            sections.append((section_type, section_text))
        else:
            logger.debug("Section '%s' has empty body, skipping", section_type)

    return sections


def sub_chunk(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> list[str]:
    """Split a large text block into overlapping chunks.

    Attempts to split on paragraph boundaries (double newline) first,
    then on sentence boundaries, then on word boundaries as a last resort.
    Each chunk is at most max_size characters. Consecutive chunks overlap
    by approximately overlap characters.
    """
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        segment = text[start:end]
        break_pos = _find_break_point(segment)

        chunk = text[start : start + break_pos].strip()
        if chunk:
            chunks.append(chunk)

        next_start = start + break_pos - overlap
        if next_start <= start:
            next_start = start + break_pos
        start = next_start

    return chunks


def _find_break_point(segment: str) -> int:
    """Find the best position to break a text segment.

    Preference order:
    1. Last paragraph break (double newline)
    2. Last sentence-ending punctuation followed by space or newline
    3. Last single newline
    4. Last space (word boundary)
    5. End of segment (hard cut)
    """
    search_start = len(segment) * 6 // 10

    # 1. Paragraph break
    pos = segment.rfind("\n\n", search_start)
    if pos != -1:
        return pos + 2

    # 2. Sentence boundary
    last_sentence = None
    for m in _SENTENCE_BREAK_RE.finditer(segment, search_start):
        last_sentence = m
    if last_sentence:
        return last_sentence.end()

    # 3. Single newline
    pos = segment.rfind("\n", search_start)
    if pos != -1:
        return pos + 1

    # 4. Word boundary
    pos = segment.rfind(" ", search_start)
    if pos != -1:
        return pos + 1

    # 5. Hard cut
    return len(segment)


def chunk_resume(
    text: str,
    candidate_name: str,
    file_name: str,
    position_tag: str | None = None,
) -> list[dict]:
    """Chunk a resume text into sections suitable for vector embedding.

    Each returned dict matches the ResumeChunk schema:
        text, section_type, chunk_index, candidate_name, file_name, position_tag

    Args:
        text: Full plaintext of the resume.
        candidate_name: Name of the candidate.
        file_name: Original uploaded file name.
        position_tag: Optional position/role tag for filtering.

    Returns:
        List of chunk dictionaries ready for embedding and storage.
    """
    if not text or not text.strip():
        logger.warning("Empty resume text for candidate '%s'", candidate_name)
        return []

    sections = detect_sections(text)
    chunks: list[dict] = []
    chunk_index = 0

    for section_type, section_text in sections:
        sub_chunks = sub_chunk(section_text, MAX_CHUNK_SIZE, OVERLAP_SIZE)

        for sc_text in sub_chunks:
            chunks.append({
                "text": sc_text,
                "section_type": section_type,
                "chunk_index": chunk_index,
                "candidate_name": candidate_name,
                "file_name": file_name,
                "position_tag": position_tag,
            })
            chunk_index += 1

    logger.info(
        "Chunked resume for '%s' (%s): %d sections -> %d chunks",
        candidate_name,
        file_name,
        len(sections),
        len(chunks),
    )

    return chunks
