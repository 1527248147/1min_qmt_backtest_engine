#!/usr/bin/env bash
#
# Publish the CURRENT TREE to GitHub as a single commit, with no history.
#
# Why this exists: same policy as the qmt_trading_scripts repo -- the current
# state is public, the development history is not. A commit, once fetched by
# anyone, cannot be recalled, so nothing is published that does not have to be.
#
# So the local repository keeps the real history (commit on master as usual;
# that is what lets you see what changed and when), and publishing builds a
# throwaway orphan branch holding one commit whose tree is exactly the
# working tree, then force-pushes it over origin/master. GitHub therefore
# never holds more than the latest state.
#
# Usage:  ./publish.sh  ["a one-line note for the snapshot"]
#
# Caveat, stated plainly: a force-push makes the previous remote commit
# unreachable, not deleted. GitHub can still serve an unreachable commit to
# anyone who knows its SHA, and a fork keeps it outright. This protects
# against browsing, cloning and search -- it is not a way to un-publish
# something that was already pushed and seen. Anything that must never be
# public has to stay out of a tracked file in the first place.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# The account. `gh` keeps several accounts in the keyring and ONE of them is
# "active"; its credential helper hands git the active account's token unless
# the remote URL names a user. On 2026-09-02 the active account flipped back to
# jieyis-tech between two publishes and the push was refused with a 403 that
# names the wrong user -- which reads like a permissions problem rather than a
# whose-token problem. Both are handled: the remote URL carries the username,
# and this switches the active account if it has drifted.
# ---------------------------------------------------------------------------
WANT_USER="1527248147"
if command -v gh >/dev/null 2>&1; then
    HAVE="$(gh api user --jq .login 2>/dev/null || echo '')"
    if [ -n "$HAVE" ] && [ "$HAVE" != "$WANT_USER" ]; then
        echo "publish.sh: gh is on '$HAVE'; switching to '$WANT_USER'"
        gh auth switch --user "$WANT_USER" >/dev/null 2>&1 || {
            echo "publish.sh: could not switch to $WANT_USER." >&2
            echo "            Run: gh auth login   (choose that account)" >&2
            exit 1
        }
    fi
fi

NOTE="${1:-}"
STAMP="$(date +%Y-%m-%d)"

if [ -n "$(git status --porcelain)" ]; then
    echo "publish.sh: the working tree has uncommitted changes." >&2
    echo "            Commit them locally first, so the snapshot you publish" >&2
    echo "            matches a commit you can actually get back to." >&2
    git status --short >&2
    exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
LOCAL="$(git rev-parse --short HEAD)"
TMP="_snapshot_$$"

# However this exits, do not leave the user parked on the orphan branch.
cleanup() {
    git checkout -q "$BRANCH" 2>/dev/null || true
    git branch -D "$TMP" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --orphan keeps the index and working tree, so this commit's tree is
# identical to what is on disk -- and `add -A` still honours .gitignore,
# which is what keeps logs/, archive/ and price_mode.txt out of it.
git checkout -q --orphan "$TMP"
git add -A
git commit -q -m "Snapshot $STAMP${NOTE:+ -- $NOTE}

Single-commit publication of the tree at local $LOCAL. This branch is
rebuilt from nothing on every publish, so the repository on GitHub carries
the current state and no prior version of it. Development history lives
only in the local clone."

git push -q -f origin "$TMP:master"

echo "published: local $LOCAL -> origin/master as a fresh single-commit snapshot"
echo "           https://github.com/1527248147/1min_qmt_backtest_engine"
