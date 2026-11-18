"""Phase D RLS — re-bind the tenant GUC at the start of every transaction.

The recording/compilation background pipelines (and the cleanup loops, which bind a
tenant per swept job) share a session across several ``commit()`` calls, and the GUC
is transaction-scoped (``set_config is_local=true``) — dropped on each commit. This
installs an ``after_begin`` listener that re-applies ``app.current_tenant`` from the
identity ContextVar at the start of EVERY transaction. Inert unless
``TENANT_RLS_MODE=enforce`` and a tenant is in context, so request first-txns
(already bound) and the no-tenant cleanup reads (which go through the SECURITY DEFINER
read resolvers) are unaffected. Scope is this process only.
"""

from __future__ import annotations

from sqlalchemy import Connection, event
from sqlalchemy.orm import Session, SessionTransaction
from w2v_common.auth.identity_context import get_tenant_id_from_context
from w2v_common.database.rls import tenant_rls_mode

_installed = False


def install_rls_tenant_rebind() -> None:
    """Register the after_begin GUC re-bind listener once per process."""
    global _installed
    if _installed:
        return

    @event.listens_for(Session, "after_begin")
    def _rebind_tenant(
        session: Session, transaction: SessionTransaction, connection: Connection
    ) -> None:
        if tenant_rls_mode() != "enforce":
            return
        tenant = get_tenant_id_from_context()
        if not tenant:
            return
        # str() at the boundary: set_config() has no uuid overload, and since the
        # tenant_id flip a tenant read back from a reaper resolver is a
        # uuid.UUID, not a str. Passing one raises
        # "function set_config(unknown, uuid, boolean) does not exist" and kills
        # the reaper sweep on every run under TENANT_RLS_MODE=enforce.
        connection.exec_driver_sql(
            "SELECT set_config('app.current_tenant', %s, true)", (str(tenant),)
        )

    _installed = True
