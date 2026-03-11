"""Resume parser service - extracts text from PDF and DOCX files."""

import io
import re

import pdfplumber
from docx import Document

from app.logger import get_logger

logger = get_logger(__name__)


def parse_pdf(content: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber.

    Args:
        content: Raw PDF file bytes.

    Returns:
        Extracted text with pages separated by newlines.

    Raises:
        ValueError: If the PDF is malformed, encrypted, or unreadable.
    """
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages_text = [
                text.strip()
                for page in pdf.pages
                if (text := page.extract_text())
            ]
    except Exception as exc:
        logger.error("Failed to parse PDF: %s", exc)
        raise ValueError(f"Failed to parse PDF file: {exc}") from exc

    return _clean_whitespace("\n\n".join(pages_text))


def parse_docx(content: bytes) -> str:
    """Extract text from DOCX bytes using python-docx.

    Args:
        content: Raw DOCX file bytes.

    Returns:
        Extracted text with paragraphs separated by newlines.

    Raises:
        ValueError: If the DOCX file is malformed or unreadable.
    """
    try:
        doc = Document(io.BytesIO(content))
        paragraphs_text = [
            text for para in doc.paragraphs if (text := para.text.strip())
        ]
    except Exception as exc:
        logger.error("Failed to parse DOCX: %s", exc)
        raise ValueError(f"Failed to parse DOCX file: {exc}") from exc

    return _clean_whitespace("\n\n".join(paragraphs_text))


async def parse_resume(file_content: bytes, filename: str) -> str:
    """Parse a resume file and return extracted text.

    Determines the file type from the filename extension and delegates
    to the appropriate parser.

    Args:
        file_content: Raw file bytes.
        filename: Original filename (used to determine file type).

    Returns:
        Extracted text from the resume.

    Raises:
        ValueError: If the file type is unsupported or parsing fails.
    """
    extension = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""

    if extension == "pdf":
        return parse_pdf(file_content)
    if extension == "docx":
        return parse_docx(file_content)

    raise ValueError(
        f"Unsupported file type: '.{extension}'. Only PDF and DOCX files are accepted."
    )


def _clean_whitespace(text: str) -> str:
    """Strip excessive whitespace while preserving section structure.

    Collapses runs of 3+ newlines down to 2 (keeping paragraph breaks)
    and trims trailing whitespace from each line.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
