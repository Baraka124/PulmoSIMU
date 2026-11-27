FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8001
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip check
COPY server.py pipeline.py synthetic.py ./
RUN useradd --create-home --uid 10001 pulmo && mkdir /app/pneumovr_jobs && chown pulmo:pulmo /app/pneumovr_jobs
USER pulmo
RUN python -c "from server import app; assert app is not None"
CMD ["sh","-c","exec python -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8001}"]
