FROM python:3.12-slim
WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ src/
COPY app.py .
COPY models/wafer_cnn.pt models/
COPY data/demo_sample.npz data/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8501"]
