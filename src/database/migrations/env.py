"""Alembic environment configuration for video_production_service."""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from src.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_sync_url() -> str:
    """Get sync database URL for migrations (convert async driver to sync)."""
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5458/video_production_service",  # pragma: allowlist secret
    )
    # Convert async driver to sync psycopg driver for migrations
    # Add sslmode if not present
    if "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=disable"
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without connecting)."""
    url = _get_sync_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect to DB and execute)."""
    url = _get_sync_url()
    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": 10},
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
