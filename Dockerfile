FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends pandoc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Data (memory, indexed documents, code graphs) lives in one volume.
ENV PAL_HOME=/data \
    PAL__SERVER__HOST=0.0.0.0
RUN useradd --create-home pal && mkdir /data /work && chown pal /data /work
USER pal
VOLUME /data
# The folder Pal sets up (its code graph, project map, documents as markdown): mount your project here.
VOLUME /work
EXPOSE 5000

# The local LLM (Ollama) runs on the host or in another container; point at it with
#   -e PAL__LOCAL_LLM__HOST=http://host.docker.internal:11434
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/healthz', timeout=2)"
CMD ["pal", "web", "--no-open", "--dir", "/work"]
