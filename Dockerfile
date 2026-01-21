FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.docker.txt /app/requirements.docker.txt
RUN pip install --no-cache-dir -r /app/requirements.docker.txt

COPY app.py /app/app.py
COPY U_Model.py /app/U_Model.py
COPY src /app/src

COPY models /app/models
COPY artifacts /app/artifacts

COPY kia_features_dict.pkl.gz /app/kia_features_dict.pkl.gz
COPY h_features_dict.pkl.gz /app/h_features_dict.pkl.gz

EXPOSE 5000
CMD ["python", "app.py"]
