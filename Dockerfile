FROM python:3.12-slim

# Dependencias del sistema para librosa / pydub / mido
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Crear directorio de instancia para SQLite y MIDI
RUN mkdir -p backend/instance backend/instance/midi_output

# Puerto dinámico que Railway inyecta
ENV PORT=8080

EXPOSE 8080

# Arrancar desde la raíz con main:app
CMD gunicorn main:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
