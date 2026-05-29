# Gunakan image Python resmi versi 3.12 yang ringan (slim) sesuai environment uv kamu
FROM python:3.12-slim

# Install uv secara langsung di dalam container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin file dependensi terlebih dahulu (memanfaatkan Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install dependensi proyek menggunakan uv secara frozen dan bersih
RUN uv sync --frozen --no-cache

# Salin seluruh kode program proyek MIRA ke dalam container
COPY . .

# Ekspos port default yang digunakan oleh Streamlit
EXPOSE 8501

# Jalankan Streamlit menggunakan environment uv yang sudah terisolasi
CMD ["uv", "run", "streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]