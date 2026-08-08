# Changelog

## [Unreleased]

### Added
- `analyze_audio` tool — native Gemma 4 audio understanding for local
  `.wav`/`.mp3` files (transcription, sound description, tone) via Ollama's
  OpenAI-compatible endpoint on an audio-capable variant (`gemma4:12b`).
  Live voice input continues to use on-device Whisper by default.
- Selectable voice transcription engine: `/voice gemma` uses Gemma 4's
  native hearing for live speech (measured ~6s/utterance vs ~0.8s Whisper);
  `/voice whisper` restores the default; unknown options now warn instead
  of being silently ignored

### Fixed
- Ctrl+C during voice recording cancels the capture instead of ending the
  session; Ctrl+C during generation cancels the turn, not the session

## [1.1.0] — 2026-08-08

### Added
- Tool approval gate: `bash_run`/`python_eval`/`write_file` require user
  confirmation in the REPL (`/confirm [on|off]`, `--yolo` to skip); the gate
  fails closed and was adversarially verified against bypass attempts
- Live token streaming — responses render token-by-token like `ollama run`
- Native Gemma 4 thinking via Ollama's `think` API — `/think [on|off]`
  controls the model directly instead of by prompt instruction
- Automatic web-grounding for prompts naming SDKs newer than the model's
  January 2025 cutoff (Google ADK, A2A, agents-cli, …) — `/ground [on|off]`
- Quoted image paths containing spaces (and trailing punctuation) are
  detected for vision input
- Output caps on every tool path — `bash_run`, `python_eval`, `read_file`,
  `list_directory`, and the ripgrep fallback — so a single command cannot
  exhaust the model's context window

### Changed
- System prompt reduced ~60%; thinking rules moved to the end for
  instruction recency; knowledge cutoff stated explicitly
- JSON tool-call parser rewritten: fence-isolated parsing, prose-brace
  tolerant scanning, bare arrays, lenient newlines in arguments
- Simple questions answer ~3x faster (measured) via concise-answer guidance

[1.1.0]: https://github.com/arjunprabhulal/gemma4-agent/releases/tag/v1.1.0

## [1.0.0] — 2026-08-08

An autonomous, multimodal (text · vision · voice) AI terminal agent powered by
Google Gemma 4 via Ollama — 100% local inference, no cloud APIs, no keys.

### Added
- Autonomous agent loop with multi-step tool calling, per-turn deduplication, and
  self-healing retries (failed calls stay retryable; successes don't loop)
- 10 built-in tools: shell, file read/write, directory listing, Python evaluator,
  web search/fetch, Agent Skill fetcher, screenshots, ripgrep search
- **Live token streaming** — responses render token-by-token like `ollama run`,
  followed by formatted markdown output
- **Automatic grounding** — prompts mentioning SDKs newer than the model's
  January 2025 cutoff (Google ADK, A2A, agents-cli, RAG Engine, …) trigger a
  visible live web search before generation, preventing fabricated APIs;
  toggle with `/ground [on|off]`
- **Vision** — image paths anywhere in a prompt are detected and sent to
  Gemma 4's native vision (paths are never stripped from your text)
- **Fully local voice mode** — VAD microphone capture, on-device Whisper
  transcription (faster-whisper `base.en`), spoken answers via macOS TTS;
  `gemma4-agent --setup-voice` pre-fetches the model like `ollama pull`
- **Agent Skills** — `fetch_skill` pulls `SKILL.md` docs from `google/skills`
  by default or any GitHub repo in the skills.sh ecosystem, with layout-agnostic
  discovery and a self-validating, poison-proof cache (`/skills clear` to reset)
- `<think>` reasoning panels with `/think [on|off]` toggle
- Slash-command REPL: `/help /tools /skills /mcp /history /model /voice /stop
  /think /ground /clear /exit` (+ aliases), with path-aware command detection
  so absolute file paths reach the model as prompts
- Experimental MCP server-config registry (`/mcp connect|disconnect|list`)
- Per-turn local telemetry: latency, prompt/completion tokens
- Test suite: 42 tests including an end-to-end REPL driver and a real-audio
  Whisper integration test

### Security
- SSRF guard on `web_fetch`: http/https only, public addresses only, validated
  IP pinned for the connection (defeats DNS rebinding), redirects re-validated
  per hop with an explicit limit
- All user/model/tool text escaped before terminal rendering (rich markup
  injection eliminated)
- Voice recordings deleted on every code path; transcription never leaves the
  machine
- Skill cache entries carry identity markers — mismatched or legacy entries are
  discarded instead of served

### Performance
- System prompt reduced ~60% (≈1.9K → ≈0.7K tokens per turn); duplicate tool
  schemas removed in favor of native function-calling definitions
- Conditional thinking and concise-answer guidance: simple queries answer ~3x
  faster (measured)
- Mixture-of-Experts default model (`gemma4:26b`, 3.8B active parameters) for
  laptop-class interactive latency

### Known limitations
- Image paths containing spaces are not detected
- No confirmation gate yet on `bash_run`/`python_eval` (top roadmap item)
- Spoken output is macOS-only (`say`); voice input works on Linux
- MCP registry stores configs only — servers are not spawned yet
- First voice use downloads the ~74MB Whisper model (once; offline after)

[1.0.0]: https://github.com/arjunprabhulal/gemma4-agent/releases/tag/v1.0.0
