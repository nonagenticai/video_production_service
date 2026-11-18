"""License token authentication middleware — delegates to w2v_common."""

# ``_looks_like_jwt`` physically lives in ``w2v_common.auth.keycloak`` but is
# imported into ``w2v_common.auth.middleware``'s namespace by the licence
# extraction rule that uses it. Re-exported here (not re-implemented) so the
# WebSocket handshake in ``src.api.ws_identity`` applies the SAME JWT-shape test
# the canonical ``AuthMiddleware._extract_license_token`` applies — a local copy
# of a three-segment check is exactly the kind of drift this seam exists to
# prevent.
from w2v_common.auth.middleware import (
    PUBLIC_ENDPOINTS,
    PUBLIC_PREFIXES,
    TRUSTED_NETWORKS,
    AuthMiddleware,
    _get_license_cache,
    _introspect_license,
    _is_trusted_network,
    _looks_like_jwt,
)

__all__ = [
    "PUBLIC_ENDPOINTS",
    "PUBLIC_PREFIXES",
    "TRUSTED_NETWORKS",
    "AuthMiddleware",
    "_get_license_cache",
    "_introspect_license",
    "_is_trusted_network",
    "_looks_like_jwt",
]
