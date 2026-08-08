"""Tests for SkillManager matching, caching, and multi-source fetching — fully offline via mocks."""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from gemma_agent.skills import SkillManager

GOOGLE_FOLDERS = [
    "agent-platform-alert-configuration",
    "alloydb-basics",
    "bigquery-ai-ml",
    "bigquery-basics",
    "cloud-logging-configuration-basics",
    "cloud-run-basics",
    "gke-ai-troubleshooting-jobset-interruption",
    "gke-basics",
]

# vercel-labs-style repo: SKILL.md folders at the repository root
COMMUNITY_SOURCE = "acme/agent-skills"
COMMUNITY_FOLDERS = ["find-skills", "nextjs", "web-design-guidelines"]


def _tree_response(paths):
    resp = Mock()
    resp.status_code = 200
    resp.json = lambda: {
        "tree": [{"path": p, "type": "blob"} for p in paths],
        "truncated": False,
    }
    return resp


def _fake_get(url, **kwargs):
    resp = Mock()
    if "api.github.com/repos/google/skills/git/trees/" in url:
        return _tree_response(
            [f"skills/cloud/{n}/SKILL.md" for n in GOOGLE_FOLDERS]
            + ["README.md", "skills/cloud/gke-basics/references/deep-dive.md"]
        )
    if f"api.github.com/repos/{COMMUNITY_SOURCE}/git/trees/" in url:
        return _tree_response([f"{n}/SKILL.md" for n in COMMUNITY_FOLDERS])
    if "raw.githubusercontent.com" in url:
        folder = url.rstrip("/").split("/")[-2]
        resp.status_code = 200
        resp.text = f"CONTENT-OF-{folder}"
        return resp
    resp.status_code = 404
    return resp


class TestSkillFetching(unittest.TestCase):

    def _fetch(self, query, cache_dir, source=None):
        sm = SkillManager(cache_dir=cache_dir)
        with patch("gemma_agent.skills.requests.get", side_effect=_fake_get):
            if source:
                return sm.search_and_fetch_github_skill(query, source=source)
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

    def test_invalid_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            res = self._fetch("anything", td, source="not a repo path")
            self.assertIn("invalid skill source", res)

    def test_community_source_root_layout(self):
        """Repos with SKILL.md folders at the root (vercel-labs style) work too."""
        with tempfile.TemporaryDirectory() as td:
            res = self._fetch("find-skills", td, source=COMMUNITY_SOURCE)
            self.assertIn("find-skills", res)
            self.assertIn("CONTENT-OF-find-skills", res)
            self.assertIn(COMMUNITY_SOURCE, res)

    def test_cache_keyed_by_matched_folder(self):
        with tempfile.TemporaryDirectory() as td:
            self._fetch("gke basics", td)
            # Cache file must carry the real skill name, not the query slug
            self.assertTrue(os.path.exists(os.path.join(td, "gke-basics.md")))
            # Second query resolving to the same folder hits the cache
            sm = SkillManager(cache_dir=td)
            with patch("gemma_agent.skills.requests.get", side_effect=_fake_get):
                res = sm.search_and_fetch_github_skill("gke-basics")
            self.assertIn("Cached Skill", res)
            self.assertIn("CONTENT-OF-gke-basics", res)

    def test_community_cache_is_source_prefixed(self):
        """Skills from different sources must not collide in the cache."""
        with tempfile.TemporaryDirectory() as td:
            self._fetch("nextjs", td, source=COMMUNITY_SOURCE)
            expected = os.path.join(td, "acme--agent-skills--nextjs.md")
            self.assertTrue(os.path.exists(expected))
            # A same-named skill in the default source would use a different file
            self.assertFalse(os.path.exists(os.path.join(td, "nextjs.md")))

    def test_poisoned_legacy_cache_is_discarded_and_refetched(self):
        """A pre-validation cache file holding the WRONG skill must never be served."""
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "gke-basics.md"), "w", encoding="utf-8") as f:
                f.write("name: alloydb-basics\nWRONG LEGACY CONTENT")
            res = self._fetch("gke-basics", td)
            self.assertIn("CONTENT-OF-gke-basics", res)
            self.assertNotIn("WRONG LEGACY CONTENT", res)
            # The refreshed cache now carries the validation marker
            with open(os.path.join(td, "gke-basics.md"), encoding="utf-8") as f:
                self.assertIn("gemma4-agent skill-cache", f.readline())

    def test_valid_cache_marker_round_trip(self):
        """A marker-validated cache entry is served without hitting the network."""
        with tempfile.TemporaryDirectory() as td:
            self._fetch("gke basics", td)
            sm = SkillManager(cache_dir=td)
            with patch("gemma_agent.skills.requests.get",
                       side_effect=AssertionError("network must not be hit on cache hit")):
                res = sm.search_and_fetch_github_skill("gke-basics")
            self.assertIn("Cached Skill", res)
            self.assertIn("CONTENT-OF-gke-basics", res)
            self.assertNotIn("skill-cache", res)  # marker stripped from output

    def test_clear_cache(self):
        import glob as glob_mod
        with tempfile.TemporaryDirectory() as td:
            self._fetch("gke basics", td)
            sm = SkillManager(cache_dir=td)
            removed = sm.clear_cache()
            self.assertGreaterEqual(removed, 1)
            self.assertEqual(glob_mod.glob(os.path.join(td, "*.md")), [])
            self.assertEqual(sm.skills, {})

    def test_no_match_lists_available(self):
        with tempfile.TemporaryDirectory() as td:
            res = self._fetch("zzz-nonexistent", td)
            self.assertIn("No matching skill found", res)
            self.assertIn("gke-basics", res)  # sample of available skills


if __name__ == "__main__":
    unittest.main()
