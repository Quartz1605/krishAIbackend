"""Utilities to normalize chatbot text for Malayalam TTS.
Removes simple Markdown emphasis markers so gTTS doesn't read 'star'.
"""
from __future__ import annotations
import re
from typing import Callable

# Patterns to remove or simplify
_EMPHASIS_RE = re.compile(r'(\*\*|__)([^\n]+?)(\*\*|__)')  # bold **text** or __text__
_ITALIC_BRACKETED_RE = re.compile(r'\*(\w[^\n]{0,200}?)\*')  # *text*
_CODE_INLINE_RE = re.compile(r'`([^`]+)`')
_MULTI_SPACES_RE = re.compile(r'\s{2,}')
_LIST_SYMBOLS_RE = re.compile(r'^[\s>*-]+', re.MULTILINE)


def strip_markdown(text: str) -> str:
    if not text:
        return text

    original = text
    # Normalize newlines
    text = text.replace('\r\n', '\n')

    # Remove bold markers
    text = _EMPHASIS_RE.sub(lambda m: m.group(2), text)
    # Remove inline code backticks (keep content)
    text = _CODE_INLINE_RE.sub(lambda m: m.group(1), text)
    # Remove simple italics
    text = _ITALIC_BRACKETED_RE.sub(lambda m: m.group(1), text)

    # Remove headings #'s and bullets, > quotes
    text = re.sub(r'^\s{0,3}[#>*\-\+]+\s*', '', text, flags=re.MULTILINE)

    # Remove remaining asterisks used as standalone tokens or emphasis remnants
    # 1. Asterisks surrounded by spaces
    text = re.sub(r'\s\*\s', ' ', text)
    # 2. Leading or trailing asterisks around words (**word**, *word*, word*)
    text = re.sub(r'\*+(\w.*?)\*+', r'\1', text)
    # 3. Any leftover isolated * or ** not attached to alphanumerics
    text = re.sub(r'(?<!\w)\*{1,2}(?!\w)', ' ', text)

    # Collapse multiple spaces and stray punctuation spacing
    text = _MULTI_SPACES_RE.sub(' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+\.', '.', text)

    cleaned = text.strip()
    if not cleaned:
        return original  # fallback to original if we stripped everything
    return cleaned

if __name__ == '__main__':  # simple manual test
    sample = "**Rubber** cultivation *guidelines*:\n- Proper `spacing` is needed."
    print(strip_markdown(sample))
