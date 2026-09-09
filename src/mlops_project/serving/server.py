from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import docker
import httpx
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[dict[str, Any]] = Field(min_length=1)
    known_covariates: list[dict[str, Any]] | None = None


class BentoRuntime:
    def __init__(self) -> None:
        self.image = os.getenv("BENTO_RUNTIME_IMAGE", "").strip()
        self.container_name = os.getenv(
            "BENTO_CONTAINER_NAME", "store-sales-forecast-bento"
        )
        self.container_port = int(os.getenv("BENTO_CONTAINER_PORT", "3000"))
        self.published_port = int(os.getenv("BENTO_PUBLISHED_PORT", "3001"))
        self.publish_host = os.getenv("BENTO_PUBLISH_HOST", "127.0.0.1")
        self.backend_host = os.getenv("BENTO_BACKEND_HOST", "127.0.0.1")
        self.health_path = os.getenv("BENTO_HEALTH_PATH", "/readyz")
        self.pull_policy = os.getenv("BENTO_PULL_POLICY", "always").lower()
        self.startup_timeout = float(os.getenv("BENTO_STARTUP_TIMEOUT_SECONDS", "120"))
        self.cleanup_on_shutdown = os.getenv(
            "BENTO_CLEANUP_ON_SHUTDOWN", "false"
        ).lower() in {"1", "true", "yes"}
        self.client = docker.from_env()
        self.container: Any | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.backend_host}:{self.published_port}"

    def _pull_image(self) -> None:
        if not self.image:
            raise RuntimeError("BENTO_RUNTIME_IMAGE must be configured")
        if self.pull_policy == "always":
            self.client.images.pull(self.image)
            return
        try:
            self.client.images.get(self.image)
        except ImageNotFound:
            self.client.images.pull(self.image)

    def _remove_existing_container(self) -> None:
        try:
            existing = self.client.containers.get(self.container_name)
        except NotFound:
            return
        existing.remove(force=True)

    def start_container(self) -> None:
        try:
            self._pull_image()
            self._remove_existing_container()
            self.container = self.client.containers.run(
                self.image,
                name=self.container_name,
                detach=True,
                ports={
                    f"{self.container_port}/tcp": (self.publish_host, self.published_port)
                },
                labels={"mlops-project.managed-by": "forecast-api"},
            )
        except (APIError, DockerException) as exc:
            raise RuntimeError(f"Could not start Bento image {self.image!r}: {exc}") from exc

    async def wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.startup_timeout
        async with httpx.AsyncClient(timeout=3) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    response = await client.get(f"{self.base_url}{self.health_path}")
                    if response.is_success:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        logs = ""
        if self.container is not None:
            logs = self.container.logs(tail=80).decode(errors="replace")
        raise RuntimeError(
            f"Bento image did not become ready within {self.startup_timeout}s.\n{logs}"
        )

    def stop(self) -> None:
        if not self.cleanup_on_shutdown:
            return
        try:
            container = self.client.containers.get(self.container_name)
            container.remove(force=True)
        except NotFound:
            pass

    async def start(self) -> None:
        await asyncio.to_thread(self.start_container)
        await self.wait_until_ready()


runtime: BentoRuntime | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runtime
    runtime = BentoRuntime()
    await runtime.start()
    try:
        yield
    finally:
        runtime.stop()
        runtime.client.close()


app = FastAPI(title="Store Sales Forecast API", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Bento runtime is not initialized")
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{runtime.base_url}{runtime.health_path}")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Bento runtime is unhealthy") from exc
    return {"status": "ok", "bento_image": runtime.image}


@app.post("/forecast")
async def forecast(payload: ForecastRequest) -> Any:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Bento runtime is not initialized")
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{runtime.base_url}/forecast",
                json=payload.model_dump(exclude_none=True),
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text[:1000],
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Bento runtime is unavailable") from exc
    return response.json()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "mlops_project.serving.server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
