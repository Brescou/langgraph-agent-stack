"""tests/test_image_extras.py — the observability extra must reach every image.

``ARG OBS_EXTRAS=""`` in infra/Dockerfile defaults to empty, so an image built
without the build-arg has no ``prometheus-client`` and serves 404 on
``/metrics``. That is issue #132: the publish job shipped such an image to GHCR
for months, the Helm ServiceMonitor scraped nothing, and the KEDA ScaledObject
had no ``active_pipelines`` series to scale on, all silently.

Three places now pass the build-arg: the publish job, the Docker smoke test,
and the Compose ``app`` service. Each hardcodes it independently, so dropping
it from one leaves the others green. These tests assert all three carry it and
agree, which is the invariant #132 was a violation of.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

OBS_BUILD_ARG = "OBS_EXTRAS=observability"
_CI_WORKFLOW = Path(".github/workflows/ci.yml")
_COMPOSE = Path("infra/docker-compose.yml")
_SMOKE = Path("tests/smoke_test_docker.sh")


def _publish_build_push_step() -> dict[str, Any]:
    """Return the publish job's docker/build-push-action step."""
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["publish"]["steps"]
    matches = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("docker/build-push-action")
    ]
    assert len(matches) == 1, f"expected one build-push-action step, got {len(matches)}"
    return matches[0]


def test_publish_job_passes_obs_extras() -> None:
    """The GHCR image must be built with the observability extra (#132)."""
    build_args = _publish_build_push_step()["with"].get("build-args", "")
    assert OBS_BUILD_ARG in build_args, (
        "the publish job builds the GHCR image without "
        f"{OBS_BUILD_ARG}, so /metrics will 404 on the published image"
    )


def test_smoke_test_builds_with_obs_extras() -> None:
    """The smoke test must exercise the same image the publish job ships."""
    text = _SMOKE.read_text(encoding="utf-8")
    assert f"--build-arg {OBS_BUILD_ARG}" in text


def test_smoke_test_follows_the_metrics_redirect() -> None:
    """``/metrics`` is a Starlette mount, so a slashless GET answers 307.

    Without ``-L`` the smoke check reads the redirect instead of the
    exposition and fails on a perfectly good image.
    """
    text = _SMOKE.read_text(encoding="utf-8")
    metrics_curl = [
        line
        for line in text.splitlines()
        if "curl" in line and "/metrics" in line and "METRICS_CODE" in line
    ]
    assert metrics_curl, "no curl call for /metrics found in the smoke test"
    for line in metrics_curl:
        flags = re.search(r"curl\s+(-\S+)", line)
        assert flags is not None, line
        assert "L" in flags.group(1), (
            "the /metrics smoke check must follow redirects (-L): a Starlette "
            f"mount answers 307 on the slashless path. Got: {line.strip()}"
        )


def test_compose_app_passes_obs_extras() -> None:
    """Local Compose must build with the extra too, as a build arg only."""
    compose = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    app = compose["services"]["app"]
    assert app["build"]["args"]["OBS_EXTRAS"] == "observability"
    # OBS_EXTRAS in environment: is a no-op; it is consumed at build time.
    environment = app.get("environment") or {}
    keys = (
        environment
        if isinstance(environment, dict)
        else dict(item.split("=", 1) for item in environment if "=" in item)
    )
    assert "OBS_EXTRAS" not in keys


def test_publish_and_compose_agree_on_the_extra_set() -> None:
    """Publish and Compose must not drift apart on which extras ship."""
    publish_args = _publish_build_push_step()["with"].get("build-args", "")
    publish_extras = {
        line.strip()
        for line in str(publish_args).splitlines()
        if line.strip() and line.strip().endswith("_EXTRAS=observability")
    }
    compose = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    compose_args = compose["services"]["app"]["build"]["args"]
    compose_extras = {
        f"{key}={value}"
        for key, value in compose_args.items()
        if key.endswith("_EXTRAS") and value == "observability"
    }
    assert publish_extras == compose_extras, (
        "the publish job and Compose disagree on the observability extra: "
        f"publish={sorted(publish_extras)} compose={sorted(compose_extras)}"
    )
