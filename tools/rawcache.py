"""Fetch a public source file once and keep it, through the vendored netcache.

Every fetch script here used to write its download straight to its final path
(`open(cache, "wb").write(raw)`), so a run killed mid-download left a
truncated CSV that the next run read as the real file. `netcache` writes to a
temporary file and renames it into place, so an entry is whole or absent.

Files now live under data/cache/<source>/<sha1>.bin. A copy already sitting at
the old data/raw/ path is adopted on first use rather than downloaded again.
"""

import pathlib

from netcache import Cache

ROOT = pathlib.Path(__file__).resolve().parent.parent


def get(source, url, user_agent, legacy=None, max_age=None, timeout=120):
    """The file's bytes, from the cache when it has been fetched before.

    `legacy` is the data/raw/ path an older version of the script saved to.
    Raises netcache.SourceError on a failed download, never returns nothing.
    """
    cache = Cache(ROOT / "data" / "cache", user_agent=user_agent)
    if legacy:
        _adopt(cache, source, url, ROOT / legacy)
    return cache.fetch(source, url, timeout=timeout, max_age=max_age)


def _adopt(cache, source, url, path):
    data_path, meta_path = cache._paths(source, url)
    if path.exists() and not data_path.exists():
        cache._write(data_path, meta_path, path.read_bytes(), url, source)
