"""
Gemma Agent Skill Manager Module.

Loads local agent skills and dynamically fetches official Google Cloud skills
live from the https://github.com/google/skills repository.
"""

import os
import glob
import re
import requests
from typing import Dict


class SkillManager:
    """Manages local and dynamic GitHub-hosted agent skills."""

    def __init__(self):
        """Initialize SkillManager and load cached/local skill files."""
        self.skills: Dict[str, Dict[str, str]] = {}
        self.cache_dir = os.path.expanduser("~/.gemma/skills")
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
                    "description": f"Skill from {filename}"
                }

    def search_and_fetch_github_skill(self, query: str) -> str:
        """
        Dynamically searches and fetches official Agent Skills live from https://github.com/google/skills repository.
        """
        query_slug = re.sub(r'[^a-zA-Z0-9]', '-', query.lower()).strip('-')
        
        # Check local cache first
        cached_file = os.path.join(self.cache_dir, f"{query_slug}.md")
        if os.path.exists(cached_file):
            try:
                with open(cached_file, "r", encoding="utf-8") as f:
                    content = f.read()
                return f"### Cached Google Skill: {query_slug}\n{content}"
            except Exception:
                pass

        try:
            # Query GitHub API for google/skills repository tree
            url = "https://api.github.com/repos/google/skills/contents/skills/cloud"
            headers = {"User-Agent": "GemmaCLI/1.0", "Accept": "application/vnd.github.v3+json"}
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code != 200:
                return f"Could not query google/skills repository (HTTP {resp.status_code})."
            
            items = resp.json()
            matched_folder = None
            
            # Find best match folder name
            for item in items:
                if isinstance(item, dict) and "name" in item:
                    foldername = item["name"]
                    if query_slug in foldername or any(q in foldername for q in query_slug.split('-') if len(q) > 2):
                        matched_folder = foldername
                        break



            if matched_folder:
                # Raw URL to SKILL.md or README.md inside the skill folder
                raw_urls = [
                    f"https://raw.githubusercontent.com/google/skills/main/skills/cloud/{matched_folder}/SKILL.md",
                    f"https://raw.githubusercontent.com/google/skills/main/skills/cloud/{matched_folder}/README.md"
                ]
                
                content = ""
                for rurl in raw_urls:
                    r_resp = requests.get(rurl, headers=headers, timeout=10)
                    if r_resp.status_code == 200:
                        content = r_resp.text
                        break

                if content:
                    # Save to local cache for fast offline access next time
                    try:
                        with open(cached_file, "w", encoding="utf-8") as f:
                            f.write(content)
                        self.skills[matched_folder] = {
                            "name": matched_folder.replace("-", " ").title(),
                            "description": f"Fetched from google/skills ({matched_folder})"
                        }
                    except Exception:
                        pass
                        
                    return f"### Official Google Skill: {matched_folder}\n{content[:4000]}"

            return f"No exact matching skill found in google/skills for '{query}'. Available skills can be searched on https://github.com/google/skills."
        except Exception as e:
            return f"Error fetching skill from google/skills repository: {str(e)}"

    def get_all_skills_system_prompt(self) -> str:
        """
        Returns summary instructions on how Gemma 4 can use fetch_google_skill to dynamically query any skill on demand.
        """
        prompt = "\n\nDYNAMIC GOOGLE SKILLS INTEGRATION (google/skills):\n"
        prompt += "You have access to the `fetch_google_skill` tool. Whenever the user requests any specialized Google Cloud, coding, AI, database, or architecture task, call `fetch_google_skill(skill_name='...')` to dynamically download the official Google skill documentation from https://github.com/google/skills!\n"
        return prompt
