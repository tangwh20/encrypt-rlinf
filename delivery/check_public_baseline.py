"""Local-only regressions for comparing with public Git objects."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from public_baseline import read_public_snapshot, select_public_merge_base


def git(directory, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    result = subprocess.run(
        ["git", "-C", str(directory), "-c", "user.name=Fixture", "-c",
         "user.email=fixture@localhost", *args], env=env, capture_output=True, check=True,
    )
    return result.stdout.decode().strip()


class PublicSnapshotTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.repo = self.root / "public"
        git(self.root, "init", "-q", str(self.repo))
        self.source = self.repo / "rlinf/module.py"
        self.source.parent.mkdir()
        self.source.write_bytes(b"value = 1\n")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "public")
        self.base = git(self.repo, "rev-parse", "HEAD")
        self.blob = git(self.repo, "rev-parse", "HEAD:rlinf/module.py")

    def test_pinned_content_survives_dirty_worktree_and_replacement(self):
        self.source.write_bytes(b"value = 2\n")
        git(self.repo, "commit", "-qam", "private")
        git(self.repo, "replace", self.base, "HEAD")
        self.source.write_bytes(b"value = 3\n")
        self.assertEqual(read_public_snapshot(self.repo, self.base),
                         (self.base, {"rlinf/module.py": b"value = 1\n"}))
        git(self.repo, "tag", "collision")
        git(self.repo, "branch", "collision")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            read_public_snapshot(self.repo, "collision")

    def test_selects_committed_fork_point_from_separate_object_stores(self):
        mirror = self.root / "公开: mirror"
        git(self.root, "clone", "--no-hardlinks", str(self.repo), str(mirror))
        git(self.repo, "commit", "--allow-empty", "-qm", "fork update")
        git(mirror, "commit", "--allow-empty", "-qm", "public update")
        public_head = git(mirror, "rev-parse", "HEAD")
        # The two post-fork commits were never exchanged.
        with self.assertRaises(subprocess.CalledProcessError):
            git(self.repo, "cat-file", "-e", public_head)
        self.assertEqual(select_public_merge_base(self.repo, mirror), (self.base, public_head))
        with self.assertRaises(subprocess.CalledProcessError):
            git(self.repo, "cat-file", "-e", public_head)
        shallow = self.root / "shallow"
        git(self.root, "clone", "--depth=1", mirror.as_uri(), str(shallow))
        with self.assertRaisesRegex(ValueError, "--public-ref"):
            select_public_merge_base(self.repo, shallow)

    def test_rejects_shared_repository_or_linked_worktree(self):
        linked = self.root / "worktree"
        git(self.repo, "worktree", "add", "--detach", str(linked), self.base)
        for other in (self.repo, linked):
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "--public-ref"):
                select_public_merge_base(self.repo, other)

    def test_rejects_multiple_merge_bases(self):
        tree = git(self.repo, "rev-parse", "HEAD^{tree}")
        left = git(self.repo, "commit-tree", tree, "-p", self.base, "-m", "left")
        right = git(self.repo, "commit-tree", tree, "-p", self.base, "-m", "right")
        first = git(self.repo, "commit-tree", tree, "-p", left, "-p", right, "-m", "merge1")
        second = git(self.repo, "commit-tree", tree, "-p", right, "-p", left, "-m", "merge2")
        git(self.repo, "update-ref", "HEAD", first)
        git(self.repo, "update-ref", "refs/heads/second", second)
        mirror = self.root / "independent.git"
        git(self.root, "clone", "--bare", "--no-hardlinks", str(self.repo), str(mirror))
        git(mirror, "symbolic-ref", "HEAD", "refs/heads/second")
        with self.assertRaisesRegex(ValueError, "found 2.*--public-ref"):
            select_public_merge_base(self.repo, mirror)

    def test_bare_promisor_cannot_fetch_missing_blob_from_local_origin(self):
        bare = self.root / "public.git"
        git(self.root, "clone", "--bare", "--no-hardlinks", str(self.repo), str(bare))
        self.assertEqual(read_public_snapshot(bare, self.base)[0], self.base)
        object_file = bare / "objects" / self.blob[:2] / self.blob[2:]
        object_file.unlink()
        git(bare, "config", "remote.origin.promisor", "true")
        git(bare, "config", "extensions.partialClone", "origin")
        with self.assertRaisesRegex(ValueError, "baseline"):
            read_public_snapshot(bare, self.base)
        self.assertFalse(object_file.exists())


if __name__ == "__main__":
    unittest.main()
