FROM python:3.10-slim

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

RUN python ingest.py

USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME=/home/user

EXPOSE 7860

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]