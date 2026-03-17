FROM python:3.10-slim

WORKDIR /app

RUN useradd -m -u 1000 user

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN python ingest.py

USER user
ENV PATH="/home/user/.local/bin:$PATH"

EXPOSE 7860

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]