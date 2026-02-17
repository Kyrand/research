# Research projects carried out by AI tools

Each directory in this repo is a separate research project carried out by an LLM tool - usually Claude Code.
Every single line of text and code was written by an LLM.

<!--[[[cog
import os
import re
import subprocess
import pathlib
from datetime import datetime, timezone

# Model to use for generating summaries
MODEL = "github/gpt-4.1"

# AI-generated note to inject into project READMEs
AI_NOTE = """
> [!NOTE]
> This is an AI-generated research report. All text and code in this report was created by an LLM.
"""

# Get the GitHub repo URL for building links
try:
    remote_url = subprocess.run(
        ['git', 'remote', 'get-url', 'origin'],
        capture_output=True, text=True, timeout=5
    ).stdout.strip()
    # Convert SSH URL to HTTPS if needed
    if remote_url.startswith('git@'):
        remote_url = remote_url.replace(':', '/').replace('git@', 'https://').removesuffix('.git')
    elif remote_url.endswith('.git'):
        remote_url = remote_url.removesuffix('.git')
except Exception:
    remote_url = ""

# Get all subdirectories with their first commit dates
research_dir = pathlib.Path.cwd()
subdirs_with_dates = []

for d in research_dir.iterdir():
    if d.is_dir() and not d.name.startswith('.'):
        # Get the date of the first commit that touched this directory
        try:
            result = subprocess.run(
                ['git', 'log', '--diff-filter=A', '--follow', '--format=%aI', '--reverse', '--', d.name],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse first line (oldest commit)
                date_str = result.stdout.strip().split('\n')[0]
                commit_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                subdirs_with_dates.append((d.name, commit_date))
            else:
                # No git history, use directory modification time
                subdirs_with_dates.append((d.name, datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)))
        except Exception:
            # Fallback to directory modification time
            subdirs_with_dates.append((d.name, datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)))

# Sort by date, most recent first
subdirs_with_dates.sort(key=lambda x: x[1], reverse=True)

# Print the heading with count
print(f"## {len(subdirs_with_dates)} research projects\n")

for dirname, commit_date in subdirs_with_dates:
    date_str = commit_date.strftime('%Y-%m-%d')
    project_url = f"{remote_url}/tree/main/{dirname}" if remote_url else dirname

    # Check for cached summary
    summary_path = research_dir / dirname / "_summary.md"
    readme_path = research_dir / dirname / "README.md"

    if summary_path.exists():
        summary = summary_path.read_text().strip()
    elif readme_path.exists():
        # Generate summary using LLM
        readme_content = readme_path.read_text()
        try:
            result = subprocess.run(
                ['llm', '-m', MODEL, '--system',
                 'Summarize this research project concisely. Write just 1 paragraph (3-5 sentences) followed by an optional short bullet list if there are key findings.'],
                input=readme_content,
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0 and result.stdout.strip():
                summary = result.stdout.strip()
                # Cache the summary
                summary_path.write_text(summary + "\n")
            else:
                summary = ""
        except Exception:
            summary = ""
    else:
        summary = ""

    # Inject AI-generated note into project README if not already present
    if readme_path.exists():
        readme_content = readme_path.read_text()
        if '<!-- AI-GENERATED-NOTE -->' not in readme_content:
            # Insert after the first # heading
            lines = readme_content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('# '):
                    lines.insert(i + 1, f"\n<!-- AI-GENERATED-NOTE -->\n{AI_NOTE.strip()}\n<!-- /AI-GENERATED-NOTE -->\n")
                    break
            readme_path.write_text('\n'.join(lines))

    print(f"### [{dirname}]({project_url}) ({date_str})\n")
    if summary:
        print(f"{summary}\n")
]]]-->
<!--[[[end]]]-->

---

To update this README locally:

```bash
pip install -r requirements.txt
cog -r -P README.md
```
