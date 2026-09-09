from __future__ import annotations

from pathlib import Path

import bentoml
import docker
from docker.utils import parse_repository_tag
from zenml import step

from pipelines.deployment.models import BentoBuildMetadata, ImageBuildMetadata


def resolve_image_tag(image_tag_template: str, *, version: str, git_sha: str) -> str:
    if "{" in image_tag_template:
        return image_tag_template.format(version=version, git_sha=git_sha)

    image_name = image_tag_template.rsplit("/", maxsplit=1)[-1]
    if ":" in image_name:
        return image_tag_template

    return f"{image_tag_template}:{version}"


def _push_image(image_tag: str) -> None:
    repository, tag = parse_repository_tag(image_tag)
    tag = tag or "latest"

    client = docker.from_env()
    try:
        for event in client.images.push(
            repository,
            tag=tag,
            stream=True,
            decode=True,
        ):
            error = event.get("error")
            if error:
                raise RuntimeError(f"Docker image push failed: {error}")
    finally:
        client.close()


@step(enable_cache=False)
def build_container_image(
    bento_archive: Path,
    bento_metadata: BentoBuildMetadata,
    image_tag: str = "mlops-project/store-sales-forecast:{version}",
) -> ImageBuildMetadata:
    bento = bentoml.import_bento(str(bento_archive))
    resolved_image_tag = resolve_image_tag(
        image_tag,
        version=bento_metadata.build_version,
        git_sha=bento_metadata.git_sha,
    )

    bentoml.container.build(
        str(bento.tag),
        backend="docker",
        image_tag=(resolved_image_tag,),
        progress="plain",
    )

    return ImageBuildMetadata(
        bento_tag=str(bento.tag),
        image_tag=resolved_image_tag,
        build_version=bento_metadata.build_version,
        git_sha=bento_metadata.git_sha,
    )


@step(enable_cache=False)
def push_container_image(
    image: ImageBuildMetadata,
    push: bool = False,
) -> ImageBuildMetadata:
    if not push:
        return image

    _push_image(image.image_tag)
    registry = image.image_tag.split("/", maxsplit=1)[0] if "/" in image.image_tag else None
    return image.model_copy(update={"pushed": True, "registry": registry})
