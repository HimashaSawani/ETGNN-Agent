with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Check if line has an unmatched odd number of quotes
    # Specifically, lines ending with literal newline inside string
    if line.endswith('"\n') and len(line) > 1 and line.strip() == '"':
        # Previous line ended without quote
        if new_lines:
            prev = new_lines.pop()
            if prev.endswith('\n'):
                prev = prev[:-1]
            new_lines.append(prev + '\\n"\n')
        i += 1
        continue
    new_lines.append(line)
    i += 1

text = "".join(new_lines)
with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "w", encoding="utf-8") as f:
    f.write(text)

print("[SUCCESS] Processed string literal line-breaks.")

