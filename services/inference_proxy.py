"""Passes the app's OpenAI-compatible API on to the llama-server running the model.

- Bodies go through as they are. llama-server implements the API itself, so the
  app only picks the server, by the request's `model`.
- Replies are passed on as they arrive, so a streamed reply streams.
- When the client goes away partway through a reply, the connection to
  llama-server is closed. That's what makes it stop generating and free the slot
  for the next request.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping

import anyio
import httpx

from models.servers import ModelServer

logger = logging.getLogger(__name__)

# Nothing else is passed on: not the app's API key, cookies or the client's host.
FORWARDED_REQUEST_HEADERS = ("content-type", "accept")


class UpstreamUnavailableError(Exception):
    """The model's server couldn't be reached, as when it has just stopped."""


class InferenceProxy:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            # Local only, so a proxy configured for the internet mustn't be used.
            trust_env=False,
            # No read timeout: on the CPU, a long prompt can take minutes before the
            # first token, and a reply that isn't streamed comes only at the end.
            timeout=httpx.Timeout(connect=5.0, read=None, write=60.0, pool=None),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
        )

    async def open(self, server: ModelServer, path: str, body: bytes, headers: Mapping[str, str]) -> httpx.Response:
        """Sends the request and returns once the reply's status and headers are
        in. Its body is read with `passthrough`, which also closes it."""
        forwarded = {name: headers[name] for name in FORWARDED_REQUEST_HEADERS if name in headers}
        # Uncompressed, so each streamed chunk can be passed on as soon as it comes.
        forwarded["accept-encoding"] = "identity"
        request = self._client.build_request("POST", server.base_url + path, content=body, headers=forwarded)
        try:
            return await self._client.send(request, stream=True)
        except httpx.TransportError as error:
            raise UpstreamUnavailableError(f"{server.model_name} isn't answering: {error}") from error

    @staticmethod
    async def passthrough(response: httpx.Response) -> AsyncIterator[bytes]:
        """The reply's body as it comes. Closing the reply, which happens here
        however it ends, closes the connection to llama-server unless the whole
        reply was read. Pass `response.aclose` to the response as a background
        task too: a client that leaves while a chunk is being sent stops this
        at the `yield`, where only that closes it at once."""
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        except httpx.HTTPError as error:
            # The server was stopped or crashed partway. The status has gone out
            # already, so all that can be done is to end the reply here.
            logger.warning("%s stopped partway through a reply: %s", response.request.url, error)
        finally:
            # A client that leaves while waiting for a chunk gets here cancelled.
            with anyio.CancelScope(shield=True):
                await response.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()


inference_proxy = InferenceProxy()
