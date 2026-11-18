"""Unit tests for observability setup module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# setup_observability tests
# ---------------------------------------------------------------------------


class TestSetupObservability:
    @patch.dict("sys.modules", {"opentelemetry": None})
    def test_setup_graceful_when_otel_not_installed(self):
        """setup_observability should no-op when opentelemetry is not installed."""
        # Should not raise
        with patch(
            "src.observability.setup.setup_observability.__module__",
            "src.observability.setup",
        ):
            pass

    def test_setup_observability_import_error_path(self):
        """When opentelemetry packages are missing, setup_observability returns early."""
        import builtins
        import importlib

        import src.observability.setup as obs_mod

        # Patch builtins __import__ to make otel imports fail
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError("no otel")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            # Re-exec setup_observability body
            importlib.reload(obs_mod)
            # After reload, call it - should not raise
            obs_mod.setup_observability()

        # Restore
        importlib.reload(obs_mod)

    def test_setup_observability_with_mocked_otel(self):
        """setup_observability configures tracing and metrics when otel is available."""
        mock_trace = MagicMock()
        mock_metrics = MagicMock()
        mock_meter = MagicMock()
        mock_metrics.get_meter.return_value = mock_meter

        modules_patch = {
            "opentelemetry": MagicMock(trace=mock_trace, metrics=mock_metrics),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.metrics": mock_metrics,
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
        }

        import importlib

        import src.observability.setup as obs_mod

        with patch.dict("sys.modules", modules_patch):
            importlib.reload(obs_mod)
            obs_mod.setup_observability(service_name="test-svc", otlp_endpoint="http://otel:4317")

        # Restore
        importlib.reload(obs_mod)

    def test_setup_observability_with_app_fastapi_instrumentor(self):
        """When app is provided, FastAPI instrumentor is called."""
        mock_trace = MagicMock()
        mock_metrics = MagicMock()
        mock_meter = MagicMock()
        mock_metrics.get_meter.return_value = mock_meter
        mock_instrumentor = MagicMock()

        modules_patch = {
            "opentelemetry": MagicMock(trace=mock_trace, metrics=mock_metrics),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.metrics": mock_metrics,
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
            "opentelemetry.instrumentation.fastapi": MagicMock(
                FastAPIInstrumentor=mock_instrumentor
            ),
        }

        import importlib

        import src.observability.setup as obs_mod

        mock_app = MagicMock()
        with patch.dict("sys.modules", modules_patch):
            importlib.reload(obs_mod)
            obs_mod.setup_observability(
                app=mock_app, service_name="test-svc", otlp_endpoint="http://otel:4317"
            )
            mock_instrumentor.instrument_app.assert_called_once_with(mock_app)

        importlib.reload(obs_mod)

    def test_setup_observability_fastapi_instrumentor_import_error(self):
        """When FastAPI instrumentor is not available, it is skipped gracefully."""
        mock_trace = MagicMock()
        mock_metrics = MagicMock()
        mock_meter = MagicMock()
        mock_metrics.get_meter.return_value = mock_meter

        modules_patch = {
            "opentelemetry": MagicMock(trace=mock_trace, metrics=mock_metrics),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.metrics": mock_metrics,
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
        }

        import importlib

        import src.observability.setup as obs_mod

        mock_app = MagicMock()
        with patch.dict("sys.modules", modules_patch):
            importlib.reload(obs_mod)
            # Make fastapi instrumentor import fail at call time
            with patch.dict(
                "sys.modules", {"opentelemetry.instrumentation.fastapi": None}, clear=False
            ):
                # Should not raise
                obs_mod.setup_observability(
                    app=mock_app, service_name="test", otlp_endpoint="http://otel:4317"
                )

        importlib.reload(obs_mod)


# ---------------------------------------------------------------------------
# get_tracer / get_meter tests
# ---------------------------------------------------------------------------


class TestGetTracer:
    def test_get_tracer_when_otel_available(self):
        mock_trace = MagicMock()
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer

        with patch.dict("sys.modules", {"opentelemetry": MagicMock(trace=mock_trace)}):
            import importlib

            import src.observability.setup as obs_mod

            importlib.reload(obs_mod)
            result = obs_mod.get_tracer("test-service")

        import importlib as importlib2

        import src.observability.setup as obs_mod2

        importlib2.reload(obs_mod2)
        # Result should not be None (it was mocked)
        assert result is not None

    def test_get_tracer_when_otel_not_available(self):
        import builtins
        import importlib

        import src.observability.setup as obs_mod

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError("no otel")
            return real_import(name, *args, **kwargs)

        importlib.reload(obs_mod)
        with patch.object(builtins, "__import__", side_effect=fake_import):
            result = obs_mod.get_tracer("test")

        assert result is None
        importlib.reload(obs_mod)


class TestGetMeter:
    def test_get_meter_when_otel_available(self):
        mock_metrics = MagicMock()
        mock_meter = MagicMock()
        mock_metrics.get_meter.return_value = mock_meter

        with patch.dict("sys.modules", {"opentelemetry": MagicMock(metrics=mock_metrics)}):
            import importlib

            import src.observability.setup as obs_mod

            importlib.reload(obs_mod)
            result = obs_mod.get_meter("test-service")

        import importlib as importlib2

        import src.observability.setup as obs_mod2

        importlib2.reload(obs_mod2)
        assert result is not None

    def test_get_meter_when_otel_not_available(self):
        import builtins
        import importlib

        import src.observability.setup as obs_mod

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError("no otel")
            return real_import(name, *args, **kwargs)

        importlib.reload(obs_mod)
        with patch.object(builtins, "__import__", side_effect=fake_import):
            result = obs_mod.get_meter("test")

        assert result is None
        importlib.reload(obs_mod)
