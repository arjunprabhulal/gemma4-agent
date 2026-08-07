"""Tests for SkillManager matching and caching — fully offline via mocks."""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from gemma_agent.skills import SkillManager

FOLDERS = [
    "agent-platform-alert-configuration",
    "alloydb-basics",
    "bigquery-ai-ml",
    "bigquery-basics",
    "cloud-logging-configuration-basics",
    "cloud-run-basics",
    "gke-ai-troubleshooting-jobset-interruption",
    "gke-basics",
]


def _fake_get(url, **kwargs):
    resp = Mock()
    if "api.github.com" in url:
        resp.status_code = 200
        resp.json = lambda: [{"name": n} for n in FOLDERS]
    else:
        folder = url.split("/skills/cloud/")[1].split("/")[0]
        resp.status_code = 200
        resp.text = f"CONTENT-OF-{folder}"
    return resp


class TestSkillMatching(unittest.TestCase):

    def _fetch(self, query, cache_dir):
        sm = SkillManager(cache_dir=cache_dir)
        with patch("gemma_agent.skills.requests.get", side_effect=_fake_get):
            return sm.search_and_fetch_github_skill(query)

    def test_exact_match_beats_earlier_token_match(self):
        with tempfile.TemporaryDirectory() as td:
            res = self._fetch("gke basics", td)
            # Previously token 'basics' matched alloydb-basics first
            self.assertIn("gke-basics", res)
            self.assertIn("CONTENT-OF-gke-basics", res)

    def test_prefix_match_prefers_shortest(self):
        with tempfile.TemporaryDirectory() as td:
            res = self._fetch("cloud-run", td)
            self.assertIn("cloud-run-basics", res)

    def test_token_match_on_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            res = self._fetch("gke", td)
            self.assertIn("gke-basics", res)

    def test_empty_slug_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            res = self._fetch("!!!", td)
            self.assertIn("No matching skill found", res)

    def test_cache_keyed_by_matched_folder(self):
        with tempfile.TemporaryDirectory() as td:
            self._fetch("gke basics", td)
            # Cache file must carry the real skill name, not the query slug
            self.assertTrue(os.path.exists(os.path.join(td, "gke-basics.md")))
            self.assertFalse(os.path.exists(os.path.join(td, "gke-basics-basics.md")))
            # Second query resolving to the same folder hits the cache
            sm = SkillManager(cache_dir=td)
            with patch("gemma_agent.skills.requests.get", side_effect=_fake_get):
                res = sm.search_and_fetch_github_skill("gke-basics")
            self.assertIn("Cached Google Skill: gke-basics", res)
            self.assertIn("CONTENT-OF-gke-basics", res)


if __name__ == "__main__":
    unittest.main()
