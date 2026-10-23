FROM python:3.10-slim

# uv installs the exact versions pinned in uv.lock (same as the model was trained with).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dependencies only: the project itself is not installed as a package,
# its code is simply copied below and found through PYTHONPATH.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY football ./football

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"
EXPOSE 8000

CMD ["uvicorn", "football.api:app", "--host", "0.0.0.0", "--port", "8000"]