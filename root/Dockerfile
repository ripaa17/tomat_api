# Gunakan image dasar dari Python
FROM python:3.9-slim

# Menetapkan direktori kerja di dalam container
WORKDIR /app

# Menyalin requirements.txt ke dalam container
COPY requirements.txt /app/

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode aplikasi ke dalam container
COPY . /app/

# Menentukan port yang akan digunakan
ENV PORT 8080

# Menjalankan aplikasi Flask
CMD ["python", "tomato.py"]
