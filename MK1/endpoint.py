import modal

from typing import List
from pydantic import BaseModel


class GenerationRequest(BaseModel):
    text: str
    max_tokens: int
    eos_token_ids: List[int] = []
    max_input_tokens: int = 0
    num_samples: int = 1
    stop: List[str] = []
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class GenerationResponseSample(BaseModel):
    text: str
    generated_tokens: int
    finished: float
    finish_reason: str


class GenerationResponse(BaseModel):
    created: float
    finished: float
    num_samples: int
    prompt: str
    prompt_tokens: int
    responses: List[GenerationResponseSample]


stub = modal.Stub(
    "mk1-endpoint-backend",
    image=modal.Image.debian_slim(),
)


@stub.function(
    keep_warm=1,
    allow_concurrent_inputs=1000,
    timeout=600,
)
@modal.asgi_app(label="mk1-chat-endpoint")
def app():
    import modal
    import fastapi
    import fastapi.staticfiles

    web_app = fastapi.FastAPI()
    Model = modal.Cls.lookup(
        "mk1-flywheel-latest-mistral-7b-instruct", "Model", workspace="mk1"
    ).with_options(
        gpu=modal.gpu.A10G(),
        timeout=600,
    )
    model = Model()

    @web_app.get("/health")
    async def health():
        stats = await model.generate.get_current_stats.aio()
        if stats.num_total_runners == 0:
            status_code = fastapi.status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            status_code = fastapi.status.HTTP_200_OK

        response = fastapi.Response(
            content="",
            status_code=status_code,
            media_type="text/plain",
        )
        return response

    @web_app.get("/stats")
    async def stats():
        stats = await model.generate.get_current_stats.aio()
        stats = {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
        }
        return stats

    @web_app.post("/generate")
    async def generate(request: fastapi.Request) -> fastapi.Response:
        content_type = request.headers.get("Content-Type")
        if content_type != "application/json":
            return fastapi.Response(
                content="",
                status_code=fastapi.status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                media_type="text/plain",
            )

        request_data = await request.json()
        generation_request = GenerationRequest(**request_data)
        response = model.generate.remote(**generation_request.dict())
        return GenerationResponse(**response)

    return web_app
