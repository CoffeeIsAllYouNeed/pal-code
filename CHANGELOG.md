# Changelog

## 0.1.5

- pal cli keeps the start screen with the logo and the chatbox (agent and model dropdowns, mic); Build is tell mode (Pal writes code), Ask is
  chat mode. Commands start with \ and are suggested in the chatbox as you type (the old /commands still work); @file loads a file.
- While Pal writes, the window shows the prompt, the reply as it streams (no code: the files are listed) and the files being built with their
  tokens and a ✅ when done. s stops, b backgrounds, j/k/d/u/g/G scroll, ↑/↓ expand or collapse the build. The prompt, the reply and the files
  built stay on screen afterwards.
- Pending changes come with a menu: (d) diffs in the browser, (g) git diff, (r) reject, (a) apply, (f) full auto, ↓ for a list, Enter to keep
  iterating. Applying shows the commands to run, runs them, rolls everything back if one fails and offers to debug it (or debugs by itself in
  full auto mode), then commits as "🤖 Pal → <message>".
- Auto modes full (the default for new plans), semi, plus, basic and none, with single settings in \config and \set-config; plans, branches,
  context, \log and \rewind (which also puts the files back), \convo, \summary, \models, \set-model, \ps, \stop and \connect.

- When a command fails, Pal asks what to do, as a choice: debug and retry once, debug in full auto mode, roll back the changes, or apply them anyway. Debugging gives the Build agent the error and this computer's system: it fixes the files, or changes the command, or adds the install step a missing tool needs, and the command runs again.
- Every yes/no question is a choice picked with the arrows (or 1, 2 ...) and Enter, with No as the safe default.
- Which model answers is never shown in the workflow lines ("Thinking" instead of the model's name; no line about routing). Characters some Windows fonts cannot draw (↳, 🏗) are replaced.
- A new terminal look: the loader is three red dots ("···", "●··", "·●·", "··●"); a build ends with a red " 🏗  Built " bar and its files as "📄 app.js ✅ | 📄 index.html ✅", then "🚀 Applied N files" and the commit; changes left pending are listed with a table of what can be done with them; plan steps are numbered with what each involves and the files it uses as red chips.
- After the first message the input is a single line under a line with the agent, the model and the keys: no box, so resizing the window no longer leaves broken lines. The header is printed once and scrolls with the conversation, so the terminal keeps its full scrollback.
- While Pal works, it writes each step as it happens and ticks it off: choosing who answers, trimming your message, picking the relevant history, documents and code, asking the model, checking the syntax, the security check, writing the files, the commit. Each line says what the step did and how long it took.
- No code in the terminal: changed files are listed one line each (new file, or lines added and removed); /diff and /review show the code.
- Build writes its changes into the folder by itself, as soon as they pass the syntax and security checks (no /apply needed); a git commit is made when the folder is a git repository, and /rewind undoes it. A critical security finding still asks first, and commands still ask before they run. To review before writing: /set auto-apply off.

## 0.1.4

- After the first message the window is really cleared (also on Windows) and the "pal code" header stays pinned at the top while the conversation scrolls under it, with the input box at the bottom.
- pal cli and pal web start at once: the screen (or the page) is there in well under a second, and the setup (Ollama, this folder's documents, map and graph, the helper, the model) runs in the background without a word. A message sent before it is ready waits for it.
- The first screen keeps the logo, "pal code" and the input box together in the middle at any window size: a smaller logo in a small window, the wordmark alone in a tiny one.
- The note about a plan open in another Pal no longer shows a process number.
- The model is chosen from a dropdown in the input box: click the model (or Ctrl+O, or /model), pick with the arrows and Enter or a click; installed models come first (under the name they are installed with), then the ones that fit your GPU, then any API model. A model that is not installed yet is downloaded when chosen.
- Each agent has its own hint in the box (Build: make me a Flappy Bird game; Plan: plan a todo app; Ask: how does the cart total work?).
- Dropdowns: short descriptions; models show only "installed" or their size, the best one for building marked "recommended"; the API choice names the providers.
- The pointer is a plain ">" (some Windows fonts have no "❯"), and Tab works again after the dropdowns were added.
- The agent has a dropdown too: click "Build agent ▾" (or /agent) and choose Build, Plan or Ask; Tab still switches.
- Under the box: just the agent and the model (no provider in brackets, no plan names), the mic as 🎤, and the hints on the right.
- While you speak, a live voice graph moves in the box (drawn with block characters every font has).
- The mic works inside the input box: tap it (or /voice), speak, press Enter or click stop, and the words are added to the box. No other screen opens.
- Faster speech to text: the speech model is loaded in the background when Pal starts, runs int8 on the CPU with one decoding pass, and takes the recording straight from memory.
- pal cli opens straight away, with no questions: click the model under the input box (or type /model) to choose another one. Only when there is no model at all yet does Pal ask for one.
- Pal chooses its helper by itself (the best small model for this computer) and downloads it when it is missing; nothing to pick.
- A mic button under the input box (click it, or Ctrl+Space): speak, and the words land in the box to edit and send. Speech to text runs on this computer (faster-whisper), installed with Pal; the web chat has its mic button too.
- No token counters on the screen.

## 0.1.2

- Choose the agent: pal cli asks Ask, Plan or Build at start (nothing preselected, saved for next time), and /agent opens the same choice any time (tab still switches).
- Tokens left: the terminal shows how much of the model's context window is left (under the input box and in the status line), and what is left of a /budget cap when one is set; the web chat bar shows "Tokens left" too.
- The start screen is just the logo, the input box and the hints: the tagline and the folder summary lines are gone, and the agent and model are shown inside the input box (/project still shows the folder summary).
- pal cli asks which models to use at every start, with nothing chosen in advance: an external API model (any provider) or a local model that suits your GPU (installed ones are marked and listed first; others are downloaded after asking), then a small local helper or none. Giving --provider/--model skips the question.
- The start screen shows the logo as solid pixels (drawn from the logo image) with a pixel wordmark "pal code" (pal white, code red); the web pages say "pal code" too. Short windows get a smaller logo, very short ones the wordmark alone.
- The input box hint is now: Build anything…  "Make me a Flappy Bird game…".
- pal bench reports input tokens (what Pal sends, its part) and output tokens (what the model writes, which varies from run to run) apart, and says when the folder is empty and there was nothing to save.
- pal bench without a task opens the usual screen; every task typed there runs once as a plain chat and once through Pal, then shows a table and bar charts (tokens and time) and offers to save Pal's version in the folder.
- The Studio (pal web) writes its build into the folder pal web was started in: straight into it when it is empty, else into a new subfolder named after the product, so nothing there is overwritten.
- Author: Vrushal More.
- The folder is shown bottom left and the Pal version bottom right; the hints under the input box are just "tab agent" and "/ commands" (? lists every shortcut).
- The repeated model line next to the folder summary is gone (only a problem with the local model is shown there).

## 0.1.1

- Fix: Pal no longer crashes on start when an optional library on the computer is broken (for example PyTorch failing with a Windows DLL error through spaCy). Such libraries are now skipped and Pal falls back to its built-in text compression.

## 0.1.0 (first release)

Pal: build and plan with AI, in your terminal (`pal cli`) or your browser (`pal web`), with fewer tokens.

- **Three agents:** Ask (answers, never changes files; asks you to choose when a good answer depends on it), Plan (numbered plans) and Build
  (file changes). Tab switches.
- **Any AI, any model:** local models through Ollama; OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mistral, Together and OpenRouter; any model
  LiteLLM supports (`litellm:<model>`); any OpenAI-compatible endpoint (`custom:<model>`); LM Studio and vLLM. Each agent can have its own model.
- **Set up for your computer:** the GPU is detected; the main model can be local (the models that suit the GPU are offered, the chosen one is
  downloaded) or an API model (then only the small helper is downloaded). No GPU: API models only. Ollama is installed and started when needed,
  always after asking.
- **Set up for your folder:** documents (PDF, Word, PowerPoint, Excel ...) become markdown, a project map lists every file's classes and
  functions, and a code graph is built with graphify; each request carries only the passages and code that match it. `PAL.md` holds the
  project's instructions (`/init`).
- **Token savings, measured:** live savings while Pal works, `/tokens` and `/cost` for the details, `pal bench` to compare against a plain chat
  on the same model.
- **A safe build workflow:** plans saved per folder, context, a sandbox for changes, `/diff` and a browser review, `/apply` with checks that
  run only after you approve them, rollback on failure, auto-debug, auto-commit, branches, rewind, autonomy levels, one terminal per plan.
- **Quality checks:** every proposed file must parse before you see it (20+ languages); `/debug <command>` fixes until a command passes;
  security checks on every change; secrets never leave the computer.
- **Like Claude Code:** prompt history, Esc to stop, Esc Esc to clear, Shift+Tab for autonomy, Ctrl+R search, `/compact`, `/btw`, `/context`,
  `/export`, `/copy`, `/resume` and more (`?` and `/help`).
- **Voice:** Ctrl+Space in the terminal and a microphone in the web chat (speech-to-text on your computer with `pal-code[voice]`, or with
  your OpenAI or Groq key). Dictation apps such as Wispr Flow work as they are.
- **Web:** `pal web` opens the chat and the Studio (plan and build small apps, with an editor) on this computer only.
