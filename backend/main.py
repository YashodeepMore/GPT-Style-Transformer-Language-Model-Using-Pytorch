import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

try:
    from .inference import TinyStoriesGenerator, UnknownTokenError
    from .schemas import GenerateRequest, GenerateResponse
except ImportError:
    from inference import TinyStoriesGenerator, UnknownTokenError
    from schemas import GenerateRequest, GenerateResponse


CHECKPOINT_PATH = Path(__file__).with_name("tinystories_model.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.generator = TinyStoriesGenerator(CHECKPOINT_PATH)
    yield
    app.state.generator = None


app = FastAPI(
    title="TinyStories Generator API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://3.109.203.136",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request):
    generator = getattr(request.app.state, "generator", None)
    return {
        "status": "ok",
        "model_loaded": generator is not None,
        "device": str(generator.device) if generator else None,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest, request: Request):
    generator = getattr(request.app.state, "generator", None)
    if generator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        stories = await asyncio.to_thread(
            generator.generate_stories,
            payload.prompt,
        )
    except UnknownTokenError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": str(exc),
                "unknown_tokens": exc.tokens,
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Story generation failed.",
        ) from exc

    if not stories:
        raise HTTPException(
            status_code=500,
            detail="The model did not generate a non-empty story.",
        )

    return GenerateResponse(success=True, stories=stories)
