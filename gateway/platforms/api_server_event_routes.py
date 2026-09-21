"""Exact public callback aliases for the API server.

``gateway.api_server.platform_event_routes`` maps one exact path to one platform.
The paths are registered on the default listener only.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

PLATFORM_EVENT_ROUTE_PATH_RE = re.compile(r"^/[A-Za-z0-9][A-Za-z0-9_/-]*$")
PLATFORM_EVENT_ROUTE_RESERVED_PREFIXES = ("/api/", "/v1/", "/p/")
PLATFORM_EVENT_ROUTE_RESERVED_PATHS = frozenset({
    "/health", "/health/detailed", "/v1/health",
})


def parse_platform_event_routes(
    raw: Any,
    *,
    native_paths: set[str],
    normalize_platform: Callable[[str], str],
) -> tuple[Dict[str, str], Optional[str]]:
    """Validate a path-to-platform mapping.

    Invalid config is an error string so startup can refuse before binding.
    An absent or empty mapping is a no-op. Stripping is applied before the
    duplicate check: ``" /hooks"`` and ``"/hooks"`` are the same path.
    """
    if raw is None or raw == {}:
        return {}, None
    if not isinstance(raw, dict):
        return {}, "platform_event_routes must be a mapping of exact path to platform"
    routes: Dict[str, str] = {}
    seen_platforms: Dict[str, str] = {}
    for path, platform in raw.items():
        if not isinstance(path, str):
            return {}, f"platform_event_routes path must be a string, got {type(path).__name__}"
        path = path.strip()
        if (
            not PLATFORM_EVENT_ROUTE_PATH_RE.fullmatch(path)
            or path.endswith("/")
            or "//" in path
        ):
            return {}, f"platform_event_routes path {path!r} is not an exact public callback path"
        if (
            path in PLATFORM_EVENT_ROUTE_RESERVED_PATHS
            or path in native_paths
            or any(path.startswith(prefix) for prefix in PLATFORM_EVENT_ROUTE_RESERVED_PREFIXES)
        ):
            return {}, f"platform_event_routes path {path!r} conflicts with a reserved API route"
        if not isinstance(platform, str):
            return {}, (
                f"platform_event_routes[{path!r}] must be a platform name string, "
                f"got {type(platform).__name__}"
            )
        platform_name = normalize_platform(platform)
        if not platform_name:
            return {}, f"platform_event_routes[{path!r}] has an invalid platform name"
        if path in routes:
            return {}, f"platform_event_routes path {path!r} is duplicated"
        prior = seen_platforms.get(platform_name)
        if prior is not None and prior != path:
            return {}, (
                f"platform_event_routes maps {prior!r} and {path!r} to the same platform "
                f"{platform_name!r}"
            )
        routes[path] = platform_name
        seen_platforms[platform_name] = path
    return routes, None
