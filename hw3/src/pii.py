"""Conservative regex masking, adapted to English issue reports.

No claim of complete personal-data detection: free-form names and addresses
require review. Ordinary version numbers and non-birth dates stay intact.
"""

import re

PATTERNS = {
    "url_credentials": re.compile(
        r"(?P<pre>\b[a-z][a-z0-9+.-]*://)[^/\s:\x22\x27<>]+:[^/@\s\x22\x27<>]+@", re.I
    ),
    "query_secret": re.compile(
        r"(?P<pre>[?&](?:token|access_token|api_key|apikey|password)=)[^&#\s\x22\x27<>]+",
        re.I,
    ),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?:\+7|\b8)[\s\-(]{0,3}\d{3}[\s\-)]{0,3}\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"),
    "phone_context": re.compile(
        r"(?P<pre>\b(?:phone|telephone|mobile|tel)\s*[:=]\s*)(?:\+?\d[\d ()-]{7,}\d)",
        re.I,
    ),
    "birth_date": re.compile(
        r"(?P<pre>(?:дат[аеы]\s+рождения|год\s+рождения|родил(?:ся|ась)|\bг\.\s?р\.|\bд\.\s?р\.|\bdate\s+of\s+birth|\bdob|\bborn\s+on)\s*[:\-—]?\s*)(?:(?:0?[1-9]|[12]\d|3[01])[.\-/](?:0?[1-9]|1[0-2])[.\-/](?:19|20)\d{2}|(?:19|20)\d{2}-\d{2}-\d{2})\b",
        re.I,
    ),
    "home_path": re.compile(r"(?P<pre>/(?:Users|home)/)[^/\s\\:]+"),
    "windows_home": re.compile(r"(?P<pre>[A-Za-z]:\\Users\\)[^\\\s:]+", re.I),
    "mention": re.compile(
        r"(?<![\w/@.])@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}[A-Za-z0-9])?(?![\w/-]|\.[A-Za-z_])"
    ),
    "signature_name": re.compile(
        r"(?P<pre>\b(?:Best regards|Kind regards|Regards|Cheers|Sincerely|Br|Thank you|Thanks)[,!.]?\s*\n\s*)(?:[A-Z][a-z]+)(?: [A-Z][a-z]+){0,2}(?=\s*(?:\n|$))"
    ),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
}
PLACEHOLDERS = {
    "url_credentials": r"\g<pre>[USER]:[PASSWORD]@",
    "query_secret": r"\g<pre>[SECRET]",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "phone_context": r"\g<pre>[PHONE]",
    "birth_date": r"\g<pre>[DATE]",
    "home_path": r"\g<pre>[USER]",
    "windows_home": r"\g<pre>[USER]",
    "signature_name": r"\g<pre>[NAME]",
    "mention": "[USER]",
    "github_token": "[TOKEN]",
}


def scrub(text):
    hits = {}
    for name, pattern in PATTERNS.items():
        if name == "mention":
            parts = re.split(r"(```[\s\S]*?```|`[^`\n]+`|[A-Za-z][A-Za-z0-9+.-]*://\S+)", text)
            count = 0
            for i in range(0, len(parts), 2):
                parts[i], found = pattern.subn(PLACEHOLDERS[name], parts[i])
                count += found
            text = "".join(parts)
        else:
            text, count = pattern.subn(PLACEHOLDERS[name], text)
        if count:
            hits[name] = count
    return text, hits
