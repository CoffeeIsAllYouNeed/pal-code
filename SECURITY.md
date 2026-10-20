# Pal security

Pal writes code into your folder and talks to an AI model, so it treats two things as untrusted: what it sends out, and what comes back.
The checks follow the [vibe-security](https://github.com/raroque/vibe-security-skill) checklist (MIT) where it applies to a coding CLI.

## Keys never leave your computer
- Secrets files (`.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa`, `credentials.json`, `.npmrc` ...) are never sent to a model, even when you `@` them. `.env.example` is allowed.
- Key-shaped strings (Stripe, OpenAI, Anthropic, AWS, GitHub, GitLab, Slack, Google keys, private key blocks, passwords in connection
  strings, `api_key = "..."`) are replaced with `[hidden secret]` in `@files` and in your own messages before anything is sent. Pal tells you when it did.
- Your API key is read from the environment for each request and is never saved: the preferences file holds only provider, model and agent.

## The model's output is untrusted
- **Nothing is written without you.** Build shows every change as a diff and writes only after you answer `y`.
- **Every proposed file is checked first:**
  - hardcoded keys
  - secret keys in client-side variables (`NEXT_PUBLIC_`, `VITE_`, `EXPO_PUBLIC_`, `REACT_APP_`)
  - a Supabase `service_role` key in front-end code
  - AI or payment APIs called from the browser
  - `eval`
  - shell commands and SQL built from strings
  - HTML inserted from variables
  - prices or roles taken from the request
  - TLS checks off, unsafe `pickle`/`yaml.load`, JWTs read without verification, CORS `*`, debug mode on, insecure random tokens
- **A critical finding changes the question** to "Write anyway?", and the answer defaults to no.
- **Paths are checked:**
  - inside the folder only: no `..`, absolute paths or symlink escapes
  - no hidden files, no secrets files
  - no programs or shell scripts (`.exe`, `.bat`, `.ps1`, `.sh` ...)
  - size and count limits
- **Nothing runs without you.** The model has no tools and no network access. After `/apply`, the checks it suggests (tests, build) run only when you answer `y` to each one; only autonomy `full` (or `/set auto-exec on`) runs them without asking, and `/set can-exec off` stops them entirely. A failed check puts every file back, and changes wait in a sandbox until you apply them.
- **Text a model writes cannot drive your terminal.** Escape sequences and control characters are stripped before anything is printed, so
  a reply or a file cannot change the window title, write to your clipboard, hide text or fake a prompt.
- **Prompt injection:** each request tells the model that file contents are data, not instructions, and adds short rules: no real keys in code,
  check prices and logins on the server, parameterised SQL, no shell or HTML built from input.

## You stay in control
- `/security` checks the folder: secrets files that are committed to git or missing from `.gitignore`, and the code checks above, worst first.
- `/security ai` asks the Plan agent for a deeper review, with the scanner's findings and the files they point at (secrets hidden first).
- `--budget 50000` or `/budget 50000` caps the tokens a session may use. Pal stops before going over. Set a spending cap with your API provider as well.
- Every file Pal writes is logged in `~/.pal/pal-audit.log`. Each entry records the time, the folder, the file, its size, a hash of the content and any findings, but never the content itself.

These are pattern checks. They catch the common mistakes quickly; they cannot prove code is safe. Review what you accept.

## Setting up the folder (pal cli, pal web)

- **Nothing is written in your folder.** The project map, the code graph and the markdown versions of your documents are kept in
  `~/.pal/projects/<folder>/`. Secrets files (`.env`, keys, certificates) and hidden folders are never read; `.palignore` leaves out more.
- **Nothing leaves the computer while setting up.** Documents are converted and the graph is built locally (graphify's AST pass, no model).
  Only the passages and excerpts that match a request are sent with it, after the same secret checks as everything else.
- **Installing Ollama asks first,** and uses Ollama's official installer (`curl -fsSL https://ollama.com/install.sh | sh` on Linux,
  `brew install ollama` on macOS, `winget install Ollama.Ollama` on Windows). Model downloads ask first too.
- **`ollama serve` is started in the background** when Ollama is installed but not running; it keeps running after Pal exits, like the
  Ollama app. Its log is `~/.pal/ollama.log`.
- **Commands Pal runs** (approved checks, `/debug`) run in your folder with your permissions, with `PYTHONDONTWRITEBYTECODE=1` so a
  file just rewritten is never run from a stale cache.
- **`pal bench`** works in temporary copies of the folder, which are deleted afterwards.

## Models, history and voice

- **Keys are never saved,** for any provider. A custom endpoint's address is saved; its key comes from `PAL_CUSTOM_API_KEY` or `/key custom`
  for the session only. A custom endpoint never receives an OpenAI key from the environment: Pal sends a placeholder unless you set one.
- **Prompt history** (`~/.pal/projects/<folder>/history`) never keeps a `/key` line, and key-shaped strings in other lines are hidden first.
- **Voice:** in the terminal, audio is recorded only between Ctrl+Space and Enter, and is not kept. In the browser it goes only to this
  computer's Pal (`/api/transcribe`), which turns it into text locally with faster-whisper, or with your OpenAI or Groq key when that is the
  engine available. Without either, the browser's own speech recognition is used (in Chrome that sends the audio to Google).
- **`/copy`** uses the system clipboard tool, or the terminal's OSC 52 sequence.

