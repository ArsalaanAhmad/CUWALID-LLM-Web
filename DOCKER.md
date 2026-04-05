# Docker Setup

## 1) Required environment variables
Create or update `.env` in the project root with at least:

- `CHAT_PROVIDER=openrouter`
- `OPENROUTER_API_KEY=<your_key>`
- `LLM_MODEL=<your_model>`
- `MODELLING_LLM_MODEL=<your_model_or_faster_model>`
- `OPENROUTER_TIMEOUT_SECONDS=18`
- `MODELLING_RETRY_COUNT=1`

## 2) Run with Docker Compose

```bash
docker compose up --build
```

App URL:

- http://localhost:8000

## 3) Stop

```bash
docker compose down
```

## 4) Optional direct Docker commands

Build:

```bash
docker build -t cuwalid-web .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env cuwalid-web
```

## Notes for partners

- Forecast data is bundled from the local `data/` folder into the image.
- If modelling is slow/unavailable, the app now returns a useful fallback response with docs instead of a hard error.
