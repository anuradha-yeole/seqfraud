FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy pandas pyarrow fastapi uvicorn pydantic
COPY seqfraud/ seqfraud/
COPY artifacts/seq_model.pt artifacts/serving_config.json artifacts/
ENV SEQFRAUD_ARTIFACTS=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "seqfraud.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
