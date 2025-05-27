import re


def remove_accent(prompt: str) -> str:
    pattern = r"Accent:(?:[^.]*\.\s*|.*?(?=[A-Z]))"
    t = re.sub(pattern, "", prompt)
    return t.strip()

