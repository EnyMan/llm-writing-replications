# Claude kit: HAP-E continuations via `claude -p`

Runs the paper's prompt on the 900 human texts of HAP-E mini (150 per genre) with Claude models through
Claude Code's headless mode, using a logged-in Claude **subscription** (`ANTHROPIC_API_KEY` is removed for the
subprocess). This is how the Claude rows in the report were produced.

Needs Python 3.9+ (standard library only) and the `claude` CLI logged in to the account you want to use.
`prompts.jsonl` is written by `../get_data.py`; the folder is self-contained, so it can be copied to another machine.

## Steps
1. Check the login: `claude` -> `/status`.
2. Sanity check (two tiny calls with Haiku):
   ```
   python run_claude.py check --models claude-haiku-5-5
   ```
   Expect `minimal` ≈ 750 input tokens and `default` ≈ 22,000 (on CLI 2.1.328; other versions differ a little).
   A much higher `minimal` means something from the machine's own Claude Code setup is being injected.
3. Pilot, 2 docs per genre (12 per model per condition):
   ```
   python run_claude.py run --n-per-genre 2
   ```
   Look at a few lines in `out/*.jsonl` (`text`, `thinking_tokens`, `is_error`).
4. Full run (900 docs x 4 models x 2 conditions = 7,200 calls; resumable, so re-run if interrupted):
   ```
   python run_claude.py run --workers 4
   ```
   One model / condition at a time: `--models claude-opus-5-5 --conditions minimal`.
   On rate limits it backs off and retries; if it gives up on some docs, re-run later.
5. Copy the `out/*.jsonl` files into `reinhart2025/out/` and run the analysis from there.

## Conditions
| condition | system prompt | tools | settings, hooks, plugins, MCP, CLAUDE.md |
|---|---|---|---|
| `minimal` | "You are a helpful assistant." | none | none |
| `default` | Claude Code's built-in prompt | built-in | none (so the machine's own setup doesn't leak in) |

Both run in an empty temp dir, with `--effort low` (same for every model; change it with `--effort`).
Claude Code still adds ~750 tokens of environment context even in `minimal`; it can't be removed
with subscription auth (`--bare` would, but it needs an API key).

## Models
`claude-haiku-5-5` (also a check against the same model via OpenRouter), `claude-sonnet-5-5`,
`claude-opus-5-5`, `claude-fable-5-2`. Change them with `--models a,b`.
