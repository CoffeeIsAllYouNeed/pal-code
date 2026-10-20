# Pal, rebuildable from scratch. The runtime is pinned (Python 3.12, Debian bookworm) and every Python package comes from
# requirements.lock with its hash, so a build today and a build next year install the same thing.
#
#   docker build -t pal-code .                                   # the app
#   docker build --target test -t pal-code-test . && docker run --rm pal-code-test   # the tests
#   docker run --rm -p 127.0.0.1:5000:5000 -v "$PWD:/work" pal-code                 # run (web on http://127.0.0.1:5000)

FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# git: Pal commits applied changes; pandoc: Word and other documents; libportaudio2: the mic (unused in a container, but importable)
RUN apt-get update && apt-get install -y --no-install-recommends git pandoc libportaudio2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock ./
RUN pip install --require-hashes -r requirements.lock
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

# -- the tests: docker build --target test -t pal-code-test . && docker run --rm pal-code-test ------------------------------------
FROM base AS test
COPY . .
RUN git config --global user.email "test@example.com" && git config --global user.name "Pal tests" \
    && rm -rf tests/studio/ui/node_modules
ENV PAL_HOME=/tmp/pal-home
CMD ["python", "-m", "pytest", "-q"]

# -- the app (the default target) ---------------------------------------------------------------------------------------------------------
FROM base AS app
# Data (memory, indexed documents, code graphs) lives in one volume.
ENV PAL_HOME=/data \
    PAL__SERVER__HOST=0.0.0.0
RUN useradd --create-home pal && mkdir /data /work && chown pal /data /work
USER pal
VOLUME /data
# The folder Pal works in: mount your project here.
VOLUME /work
WORKDIR /work
EXPOSE 5000
# A local model (Ollama) runs on the host or in another container: -e PAL__LOCAL_LLM__HOST=http://host.docker.internal:11434
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/healthz', timeout=2)"
CMD ["pal", "web", "--no-open", "--dir", "/work"]
