"""api/endpoints/packs.py — Pack discovery and traffic-weight management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from pack_kernel.registry import PackRegistry

router = APIRouter(tags=["packs"])


@router.get("/packs", summary="List registered domain packs", response_model=list[dict])
async def list_packs() -> list[dict[str, Any]]:
    """Return all registered domain packs with their input/output JSON schemas."""
    return PackRegistry.list_packs_with_metadata()


@router.get("/packs/{pack_id}/versions", summary="List versions of a registered pack")
async def list_pack_versions(pack_id: str) -> list[dict[str, Any]]:
    """Return all registered versions for a pack with their current weights."""
    try:
        versions = PackRegistry._get_versions(pack_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Pack '{pack_id}' not found."
        )
    return [{"version": pv.version, "weight": pv.weight} for pv in versions]


@router.patch(
    "/packs/{pack_id}/versions/{version}/weight",
    summary="Update traffic-split weight for a pack version",
)
async def update_pack_version_weight(
    pack_id: str,
    version: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Set the traffic-split weight for a specific registered pack version."""
    weight = body.get("weight")
    if weight is None or not isinstance(weight, (int, float)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'weight' field (number) is required.",
        )
    try:
        PackRegistry.set_weights(pack_id, {version: float(weight)})
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return {"pack_id": pack_id, "version": version, "weight": float(weight)}
