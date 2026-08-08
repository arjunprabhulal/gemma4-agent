"""
Gemma Agent Skill Manager Module.

Loads local agent skills and dynamically fetches Agent Skills from any GitHub
repository following the SKILL.md convention (the format indexed by skills.sh).
Defaults to Google's official https://github.com/google/skills repository.
"""

import os
import glob
import re
import requests
from typing import Dict, Optional

DEFAULT_SOURCE = "google/skills"
_SOURCE_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HEADERS = {"User-Agent": "GemmaCLI/1.0", "Accept": "application/vnd.github.v3+json"}


class SkillManager:
    """Manages local and dynamic GitHub-hosted agent skills."""

    def __init__(self, cache_dir: str = None):
        """Initialize SkillManager and load cached/local skill files.

        Args:
            cache_dir: Override the skill cache location (used by tests to
                avoid touching the user's real ~/.gemma).
        """
        self.skills: Dict[str, Dict[str, str]] = {}
        self.cache_dir = cache_dir or os.path.expanduser("~/.gemma/skills")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.load_local_skills()

    def load_local_skills(self) -> None:
        """
        Scan `./skills/*.md` and `~/.gemma/skills/*.md` for downloaded skills
        and index them by name for the `/skills` listing.
        """
        search_paths = [
            os.path.join(os.getcwd(), "skills", "*.md"),
            os.path.join(self.cache_dir, "*.md")
        ]
        for pattern in search_paths:
            for filepath in glob.glob(pattern):
                filename = os.path.basename(filepath)
                skill_id = os.path.splitext(filename)[0].lower()
                self.skills[skill_id] = {
                    "name": skill_id.replace("-", " ").title(),
                    "description": f"Cached skill doc ({filename}), injected on demand via fetch_skill"
                }

    def _cache_filename(self, source: str, skill: str) -> str:
        # Default-source cache keeps plain names for backwards compatibility;
        # other sources are prefixed so skills from different repos can't collide.
        if source == DEFAULT_SOURCE:
            return f"{skill}.md"
        return f"{source.replace('/', '--')}--{skill}.md"

    @staticmethod
    def _cache_marker(source: str, skill: str) -> str:
        return f"<!-- gemma4-agent skill-cache source={source} skill={skill} -->"

    def _read_cache(self, source: str, skill: str) -> Optional[str]:
        """Return cached content only if its embedded marker proves it is the
        right skill from the right source. Unmarked/mismatched files (including
        anything written by older versions with the wrong-match bug) are
        deleted so the next fetch replaces them with verified content."""
        path = os.path.join(self.cache_dir, self._cache_filename(source, skill))
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline().rstrip("\n")
                rest = f.read()
            if first == self._cache_marker(source, skill):
                return rest
            os.remove(path)  # stale/poisoned entry — force a fresh fetch
        except Exception:
            pass
        return None

    def clear_cache(self) -> int:
        """Delete all cached skill files and reset the in-memory index."""
        removed = 0
        for filepath in glob.glob(os.path.join(self.cache_dir, "*.md")):
            try:
                os.remove(filepath)
                removed += 1
            except Exception:
                pass
        self.skills = {}
        self.load_local_skills()
        return removed

    @staticmethod
    def _rank_match(query_slug: str, names) -> Optional[str]:
        """Exact name, then shortest prefix match, then most '-'-token overlaps.

        First-substring-match previously returned alphabetically-early wrong
        skills ('gke basics' -> alloydb-basics), so ranking is deliberate.
        """
        names = list(names)
        if query_slug in names:
            return query_slug
        prefixed = sorted((n for n in names if n.startswith(query_slug)), key=len)
        if prefixed:
            return prefixed[0]
        tokens = [t for t in query_slug.split('-') if len(t) > 2]
        best_score, best = 0, None
        for n in names:
            parts = n.split('-')
            score = sum(1 for t in tokens if t in parts)
            if score > best_score:
                best_score, best = score, n
        return best

    def search_and_fetch_github_skill(self, query: str, source: str = DEFAULT_SOURCE) -> str:
        """Fetch an Agent Skill's SKILL.md from a GitHub repository.

        Works with any repo following the SKILL.md convention (google/skills,
        vercel-labs/skills, ...): the repo tree is scanned for SKILL.md files
        wherever they live, so no folder layout is assumed.
        """
        source = (source or DEFAULT_SOURCE).strip().strip("/")
        if not _SOURCE_RE.match(source):
            return f"Error: invalid skill source '{source}' — expected GitHub 'owner/repo' format."

        query_slug = re.sub(r'[^a-zA-Z0-9]', '-', query.lower()).strip('-')
        if not query_slug:
            return f"No matching skill found for '{query}'. Try a specific name like 'gke-basics'."

        # Exact-name cache hit (cache files are keyed by the REAL skill name,
        # so a hit here is guaranteed to be the right skill).
        cached = self._read_cache(source, query_slug)
        if cached is not None:
            return f"### Cached Skill ({source}): {query_slug}\n{cached[:4000]}"

        try:
            # One tree call finds every SKILL.md regardless of folder layout.
            tree_url = f"https://api.github.com/repos/{source}/git/trees/HEAD?recursive=1"
            resp = requests.get(tree_url, headers=_HEADERS, timeout=10)
            if resp.status_code != 200:
                return f"Could not query the {source} repository (HTTP {resp.status_code})."

            data = resp.json()
            skill_paths: Dict[str, str] = {}
            for entry in data.get("tree", []):
                path = entry.get("path", "")
                if entry.get("type") == "blob" and path.endswith("SKILL.md"):
                    parent = os.path.basename(os.path.dirname(path)).lower()
                    name = parent or source.split("/")[1].lower()
                    # Prefer the shallowest path when names collide
                    if name not in skill_paths or path.count("/") < skill_paths[name].count("/"):
                        skill_paths[name] = path

            if not skill_paths:
                return f"No SKILL.md files found in {source}."

            matched = self._rank_match(query_slug, skill_paths.keys())
            if not matched:
                sample = ", ".join(sorted(skill_paths)[:10])
                return f"No matching skill found in {source} for '{query}'. Available include: {sample}"

            # Matched-name cache hit (cache is keyed by the matched skill,
            # never the query — a query-keyed cache mislabeled wrong matches).
            cached = self._read_cache(source, matched)
            if cached is not None:
                return f"### Cached Skill ({source}): {matched}\n{cached[:4000]}"

            raw_url = f"https://raw.githubusercontent.com/{source}/HEAD/{skill_paths[matched]}"
            r_resp = requests.get(raw_url, headers=_HEADERS, timeout=10)
            if r_resp.status_code != 200:
                return f"Found skill '{matched}' in {source} but could not download it (HTTP {r_resp.status_code})."
            content = r_resp.text

            # Save to local cache for fast offline access next time, with a
            # marker header so reads can verify it is the right skill.
            try:
                cache_path = os.path.join(self.cache_dir, self._cache_filename(source, matched))
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(self._cache_marker(source, matched) + "\n" + content)
                self.skills[matched] = {
                    "name": matched.replace("-", " ").title(),
                    "description": f"Fetched from {source} ({matched})"
                }
            except Exception:
                pass

            return f"### Skill from {source}: {matched}\n{content[:4000]}"
        except Exception as e:
            return f"Error fetching skill from {source}: {str(e)}"

    def get_all_skills_system_prompt(self) -> str:
        """
        Returns instructions on using fetch_skill to pull Agent Skills on demand.
        """
        prompt = (
            "\n\nAGENT SKILLS: `fetch_skill(skill_name=...)` pulls official docs from google/skills "
            "into context; pass source='owner/repo' for community SKILL.md repos "
            "(discoverable via `npx skills find <topic>` through bash_run when npx exists). "
            "Fetch only for product details you are unsure about.\n"
        )
        return prompt
