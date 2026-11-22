"""The proxy against a stand-in transport, to see exactly what it sends and
when it closes the connection to llama-server."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import anyio
import httpx
import pytest
from starlette.datastructures import Headers

from models.servers import ModelServer
from services.inference_proxy import InferenceProxy, UpstreamUnavailableError

SERVER = ModelServer(
    id="abc123",
    repo_id="unsloth/Qwen3.8-27B-GGUF",
    quantization="Q4_K_M",
    model_path="C:/models/q4.gguf",
    runtime_id="llama-b11960-bin-win-rocm-10.0-x64",
    backend="rocm",
    host="127.0.0.1",
    port=8080,
    base_url="http://127.0.0.1:8080",
    model_name="unsloth/Qwen3.8-27B-GGUF:Q4_K_M",
    log_path="C:/logs/abc123.log",
    created_at=datetime.now(UTC),
)


class Body(httpx.AsyncByteStream):
    """A reply's body: these chunks, then `then`, which is "end", "hang" or an
    error to raise. It records being closed."""

    def __init__(self, chunks: list[bytes], then: str | Exception = "end") -> None:
        self.chunks = chunks
        self.then = then
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.then == "hang":
            await anyio.sleep_forever()
        elif isinstance(self.then, Exception):
            raise self.then

    async def aclose(self) -> None:
        self.closed = True


def proxy_for(handler: Callable[[httpx.Request], httpx.Response]) -> InferenceProxy:
    return InferenceProxy(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def opened(body: Body) -> httpx.Response:
    """The reply as the proxy gets it, its body still to read."""

    async def open_it() -> httpx.Response:
        proxy = proxy_for(lambda request: httpx.Response(200, stream=body))
        return await proxy.open(SERVER, "/v1/chat/completions", b"{}", Headers())

    return anyio.run(open_it)


def test_only_the_body_and_its_type_go_on() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    async def send() -> None:
        proxy = proxy_for(handler)
        headers = Headers(
            {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Authorization": "Bearer app-secret",
                "Cookie": "session=1",
                "Host": "192.168.1.20:8000",
            }
        )
        response = await proxy.open(SERVER, "/v1/chat/completions", b'{"model": "x",  "a": 1}', headers)
        await response.aclose()

    anyio.run(send)
    [request] = seen
    assert (request.method, str(request.url)) == ("POST", "http://127.0.0.1:8080/v1/chat/completions")
    assert request.content == b'{"model": "x",  "a": 1}'
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["accept-encoding"] == "identity"
    assert request.headers["host"] == "127.0.0.1:8080"
    assert "authorization" not in request.headers and "cookie" not in request.headers


def test_a_server_that_doesnt_answer_is_unavailable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async def send() -> None:
        await proxy_for(refuse).open(SERVER, "/v1/chat/completions", b"{}", Headers())

    with pytest.raises(UpstreamUnavailableError, match=f"{SERVER.model_name} isn't answering"):
        anyio.run(send)


def test_the_reply_goes_back_chunk_by_chunk_and_is_closed() -> None:
    body = Body([b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"])
    response = opened(body)

    async def read() -> list[bytes]:
        return [chunk async for chunk in InferenceProxy.passthrough(response)]

    assert anyio.run(read) == [b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"]
    assert body.closed


def test_a_client_that_stops_reading_closes_the_reply() -> None:
    body = Body([b"data: 1\n\n"], then="hang")
    response = opened(body)

    async def read_one() -> None:
        chunks = InferenceProxy.passthrough(response)
        assert await anext(chunks) == b"data: 1\n\n"
        await chunks.aclose()

    anyio.run(read_one)
    assert body.closed


def test_a_client_that_leaves_while_waiting_closes_the_reply() -> None:
    """How a disconnect arrives: the request's task is cancelled."""
    body = Body([b"data: 1\n\n"], then="hang")
    response = opened(body)

    async def leave() -> None:
        first = anyio.Event()

        async def read() -> None:
            async for _ in InferenceProxy.passthrough(response):
                first.set()

        async with anyio.create_task_group() as group:
            group.start_soon(read)
            await first.wait()
            await anyio.sleep(0.05)
            group.cancel_scope.cancel()

    anyio.run(leave)
    assert body.closed


def test_a_server_that_dies_partway_ends_the_reply() -> None:
    body = Body([b"data: 1\n\n"], then=httpx.RemoteProtocolError("peer closed connection"))
    response = opened(body)

    async def read() -> list[bytes]:
        return [chunk async for chunk in InferenceProxy.passthrough(response)]

    assert anyio.run(read) == [b"data: 1\n\n"]
    assert body.closed
