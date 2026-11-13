# The AI assistant

The FlexTool AI assistant answers questions about a FlexTool project: what a parameter is set to in a scenario,
what a result table shows, why a run failed, where slack appeared, how two scenarios differ. It reads what it needs
with a set of read-only FlexTool tools. The tools cannot change input data, settings or results, and they cannot
start a run.

The tools give the assistant:

- the input data of the project's input sources, scenario by scenario, as FlexTool would use it (the winning value
  of each parameter after the alternatives are ranked);
- the results of the scenarios that have been run (the parquet files in `output_parquet/`), with filters, sums and
  averages over the dimensions that allow them, and comparisons between scenarios;
- the run log and run status of each run, including which solves were accepted and the slack (penalty) in the
  results;
- the FlexTool documentation, including the model's equations and the list of result tables.

There are three ways to use it:

- **Chat in the FlexTool window**: "AI assistant…" in the side menu opens a chat window for the open project. It
  talks to an AI model of the service you choose (a service on the internet with your key, or a model running on
  this computer), gives it the tools, and tells it what you have selected in the window.
- **Chat in the terminal**: `flextool-assistant` does the same in a terminal.
- **Connect an AI agent** such as Claude Code: the agent starts the tools itself and uses its own model.

The window and the terminal share the settings, the keys, the accepted data notices and the saved conversations.

## Where your data goes

Your questions and everything the tools return — input data, results, run logs and documentation, the names of
your project and scenarios, and the model facts of the chosen scenario (its solves, periods, time steps and latest
run, see "Ask") — are sent to the AI service that answers. FlexTool cannot say how that service stores
or uses the data: it may be kept, reviewed or used for training, depending on the service, your account and your
plan. Free plans often let the provider use your data for training. Check the provider's current policies yourself
before you use a project with data you may not share. FlexTool takes local folder paths, your user name and the
computer's name out of what the tools return, but it does not remove anything else: names of units, nodes and
scenarios, and all values, are sent as they are.

The only reliable way to keep the data on this computer is a model that runs on it (Ollama or LM Studio), which
needs a powerful computer. Some models offered through such programs still run elsewhere: Ollama's cloud models
are sent to Ollama's servers. The chat finds out before it sends anything and treats them as a service on the
internet.

## Chat in the FlexTool window

### Open the chat

"AI assistant…" is in the side menu above "Execution jobs…"; the ⚙ button beside it opens the assistant settings.
The first time either is used, a short text explains how the assistant works ("How this works" in the chat window
shows it again); "Open settings" there goes straight to the settings. Opening FlexTool or the chat window sends
nothing: your data leaves this computer only when you send a question.

The chat window belongs to the open project. It closes when you switch to another project, rename the project or
delete it, and it is as tall as the screen it opens on. A size and place you give it are remembered for that screen
layout ("Reset window layout" in "UI settings…" forgets them).

### Settings

| Setting | What it does |
|---|---|
| Provider, Address | The AI service (see [Services](#services)); the address can be changed for a model on this computer or a custom endpoint. |
| Key | "Store key" keeps the pasted key in the system keyring, or, after a warning, unencrypted in a file. A key in the service's environment variable is used instead of a stored one, and the dialog says so. The key is never shown again. |
| Model | The service's own model list ("Refresh list"), or type a model name. The list shows only chat models, and for a remote service of them only stable releases, the newest version of each model (Ollama and LM Studio list every chat model you installed); "Show all models" — or `--all` with `--list-models` in the terminal — also shows pre-releases (preview, experimental, beta…), `-latest` aliases, older versions and models whose name says they are for something else than chat (embedding, tts, transcribe, image, video, music, live, realtime and the like). A chosen model the short list leaves out stays in it, marked "(pre-release)", "(alias)", "(older version)" or "(not a chat model)". "Test model" sends one tiny request (no data of your project; after the data notice) and says whether the model answers, with the time to the first part of the answer and to its end, or the provider's own words when it refuses; it waits up to 300 seconds and the button stops it meanwhile. A model the provider says is not available to your account is then left out of the list and marked "(not available to your account)" under "Show all models", until Refresh list, a new key, a successful test or 7 days. The mark holds only for the key the test used (FlexTool saves a one-way fingerprint of the key, never the key), so a key changed in the environment variable, the keyring or the key file does not see it. In the terminal: `flextool-assistant --test-model --provider ID --model NAME`. |
| Level | Automatic (from the model name and its context size), or a lower level you choose; the list offers only the levels the model allows, and the lines below it say why (see [Assistant levels](#assistant-levels)). For a model FlexTool does not recognise, higher levels are offered at your own risk. |
| Free plan | Fewer steps for each question, to stay within the request limits of a free plan: Automatic (on only for an OpenRouter model whose name ends `:free`), On or Off (see [The free plan](#the-free-plan)). |
| Questions answered at the same time | 1 to 4 (default 1): how many chats may wait for the service at once; more chats wait their turn, so several chats on a free plan do not multiply its "too many requests" answers. |
| Data use | Where the data goes, the privacy settings sent with every request (OpenRouter's data-collection setting can be turned off, after a question), and the settings only your account can change. |
| Keep conversations for | Days (default 30). Older conversations are deleted when FlexTool closes; `0` saves no conversations. |

Saved settings apply to each chat from its next question, except the provider, the address and the model: a chat
keeps the model that answered in it, also when it is reopened, and with it its level (unless you change the level
of that same model). A new chat, or one without an answer yet, uses the saved ones. The top of the window shows the
selected chat's service, model and level ("Anthropic · claude-opus-5-5 · advanced") and where its data goes.

### Ask

Type in the box at the bottom and press Enter to send; Ctrl+Enter starts a new line. The question shows at once,
and you can type the next one while the answer comes; Send is available again when the answer is done. Stop (or
Escape) stops the answer; what arrived stays in the chat.

Each question you sent has two small buttons after it: ↻ asks it again as it was, and ✎ puts it in the box to
change it (where the symbols cannot be shown, on Windows without the Segoe UI Symbol font, they read "Ask again" and
"Edit"). R or E on a question line, and its right-click menu, do the same. The question's answer, and every later
question with its answer, are removed from the chat and from what the assistant is sent next; when later questions
go too, the window asks first ("This also removes the 2 later questions and answers.", a question still being
answered counted). R always asks first ("Ask question 2 again? Its answer is removed."), so a stray key press
removes nothing. During an answer, the answer stops once you confirm, and the question goes right after. With ✎
the status line says "Editing question N: Enter sends it, Escape cancels."; nothing is removed until you press
Enter, and Escape gives the box back what you had typed in it (while an answer is coming, Escape stops the answer
instead; press it again to cancel the edit). If the question you are changing is removed meanwhile (you asked an
earlier one again), the edit ends and the box gets back what you had typed. A question asked again goes to the
chat's own model, also the first one. A chat started with a summary of an earlier one (*New chat with summary*)
keeps the summary hidden and sends it with its first question again. Results the assistant had already shortened
to keep the conversation short stay shortened. The token line's "Whole chat" still counts the tokens of the removed answers:
they were used.

Each question also tells the assistant what you have selected in the FlexTool window: the input sources and
scenarios checked and highlighted, the results checked and highlighted (with whether their latest run succeeded),
and what the result viewer shows (mode, scenarios, plot and time range). The assistant uses the highlighted input
source and scenario when your question names none.

When a scenario is chosen (the highlighted one in the window, `--scenario` in the terminal), each question also
gives the assistant the scenario's **model facts**, read from the input data and the scenario's latest run, so it
does not have to guess them: the solves in their order (nested ones too) and how each solves (one solve, a rolling
window with its horizon and jump, stochastic forecast branches with their weights, representative periods), the
periods with the years each represents and which of them a solve does not realise (look-ahead periods, and periods
whose investments only it realises; with solves of different periods, also the periods realised across all solves),
the length of the time steps each solve really uses (with `new_stepduration`, the aggregated steps), the share of a
year each realised period covers (from the result table `period_share_of_year__d`), and whether the latest run
succeeded, with its error when it did not. The annualised per-period (`_d_`) results (flows, energy, annualised
costs, emissions, start-ups) are, per realised period, its weighted sum over its time steps divided by that share;
capacities, prices, averages, ratios and discounted costs are not scaled. A stochastic solve without a realized
branch is said to have empty operational results. When the time steps are not all of the same length, the facts say so
and how to deal with it. Without a chosen scenario there are no model facts. The facts are short (at most about
200 tokens) and are sent again only when they change.

Each tool call is one line in the chat; click it (or press Enter on it) to see the arguments and the first 4 000
characters of what was sent to the service. Under the buttons, always in view, a line gives the tokens of the last
question and of the whole chat, rounded to thousands ("This question: 26k in, 8k out · Whole chat: 250k in, 5k
out"; "in" counts every token sent, also the ones the service read from its cache), and the window's last line
gives the service's tokens today and this month, rounded the same way ("Anthropic today 12k in, 1k out · this month
256k in, 9k out (this FlexTool installation)"); today and the month follow your computer's calendar, and both count every chat and terminal session of this FlexTool installation. An error that a setting fixes
comes with an "Open settings" button; a conversation that got too long for the model offers "New chat" and "New chat
with summary".

### Data notice and where the data goes

The top line of the window says where the data of the current chat goes, in the same words as the terminal (see
[Start a chat](#start-a-chat)), and a second line says when a privacy setting is turned off. Before the first
question goes to a service on the internet or to an Ollama cloud model, the window shows the data notice of
[The data notice](#the-data-notice) with "Send to …" and "Cancel"; Cancel sends nothing. "Send to …" cannot be
pressed in the first half second, so a key pressed while you were still typing does not accept it. Accepting it
in the window or in the terminal is the same acceptance.

### Chats

Each chat is a tab, named by a number ("1", "2", …) that a chat of the project keeps for good. The last tab is
always an empty "New chat": sending a question there makes it a chat, and a new "New chat" appears. Several chats
can answer at the same time (see "Questions answered at the same time"). A sign before the number says what a chat
is doing: "…" working, "‖" waiting for its turn, "?" its data notice is open, "!" its last question ended with an
error, "•" an answer arrived while you were in another chat.

- **Chats menu**: the open chats (also those that do not fit in the tab row, counted on the button as "Chats (3
  more)"), New chat, "Reopen a saved chat", "Delete a saved chat" and "Delete all closed chats…", Rename chat… and
  Close chat. A saved chat is listed with its number, its name, its date and the first words of its first
  question.
- **Rename**: "Rename chat…" in the Chats menu or the tab's right-click menu, or a double click on the tab. An empty
  name shows the number again.
- **Close**: "Close chat", Ctrl+W or a middle click. The answer stops, what was received is saved, and the chat is
  under "Reopen a saved chat". When conversations are not saved (`0` days), closing a chat that has questions asks
  first.
- **Keys**: Ctrl+Tab and Ctrl+Shift+Tab (or Ctrl+Page Down and Ctrl+Page Up) go to the next and the previous chat,
  Ctrl+T to "New chat". On macOS Command+T and Command+W work too.

Answers show their headings, lists, **bold**, *italic*, `code` and tables. Bold and italic text has the size and
font of the rest of the answer, and code the fixed-width font at the same size; headings are larger. The text of
the chat window (the answers, the box you type in, the tool lines) and of the assistant's dialogs is one point
larger than FlexTool's body text: 11 pt at the default size, and it follows the font size setting.

The chats that were open come back when you open the chat window again, also after FlexTool restarts. An answer
that was in progress is not restarted; ask again if you need it.

Closing FlexTool stops every answer without asking and saves what was received; a tool call that was running is
recorded as not run, so the chat can go on after a restart. Renaming a project moves its saved chats with it, and
deleting a project deletes them. While a chat is still finishing a tool call, a rename or deletion of its project
is refused for a moment: "The AI assistant is still finishing a tool call. Try again in a moment."

## Chat in the terminal

### What you need

- FlexTool installed (see [Install with FlexTool GUI](install_gui.md) or [Install with Spine Toolbox](install_toolbox.md)).
  After an update, run "Update FlexTool" (or `pip install -e .` in a source checkout) so that the
  `flextool-assistant` command exists.
- An account with an AI service and its key, or a model running on this computer (see
  [Models on this computer](#models-on-this-computer)). The model must support tool calls: the assistant reads
  your data with them.
- A FlexTool project folder: the folder of a project of the FlexTool window (`projects/<project name>/`), or any
  folder with an `input_sources/` folder and the result folders FlexTool writes.

### Start a chat

```bash
flextool-assistant --project-path /home/me/flextool/projects/MyProject
```

The first time, the chat prints a short text on how the assistant works. Then, if no AI service is set up yet, it
asks for one: the service, its address (for a model on this computer or a custom endpoint), the key (typed without
being shown, then stored in the system keyring) and a model from the service's own list. It offers to keep these
for the next start. Starting the chat sends none of your data; only the service's model list is fetched while you
choose.

Then type a question after `you>`. The answer appears as it arrives. Each tool call the assistant makes is one dim
line, for example `· search_docs query="penalty_up" → 5 results (0.1 s)`, and after each answer a line gives the
tokens used, rounded to thousands (`tokens 26k in (20k from cache) · 8k out · conversation 250k in · 5k out`). End
a line with `\` to go on with the question on the next line. Ctrl-C stops an answer at once;
Ctrl-C or Ctrl-D at the prompt leaves the chat.

The start of the chat shows where the data goes:

- "Data goes to: Google Gemini (generativelanguage.googleapis.com)." for a service on the internet, with " It is
  sent through the proxy {proxy}." when your proxy settings send it through a proxy;
- "Endpoint on this computer (localhost:11434). FlexTool sends data only there; if that server forwards requests,
  data leaves this computer." for a model server on this computer;
- "Endpoint on your local network (192.168.1.20)." for a server on your local network;
- "Data goes to: Ollama cloud (ollama.com), forwarded by Ollama on this computer." for an Ollama cloud model
  ("forwarded by Ollama at 192.168.1.20:11434." when that Ollama runs elsewhere on your network).

For an address that starts with `http://` and is neither on this computer nor on your local network, the line
ends with " The connection is not encrypted: the key and the data can be read on the way." Use an `https://`
address for such a server when it has one.

### The data notice

Before the first question goes to a service on the internet (or to Ollama's cloud), the chat shows the data notice:
what is sent, to which service and address, that FlexTool cannot guarantee how it is stored or used, and, when a
proxy applies, that the proxy can also see it (and, for an unencrypted `http://` address on the internet, that
the connection is not encrypted). Type `yes` to send; anything else sends nothing. The answer is kept,
so the notice is shown again only when the service, its address, the proxy or the notice itself changes. A model on
this computer or your local network needs no notice.

### One question from a script

```bash
flextool-assistant --project-path /home/me/flextool/projects/MyProject -q "Which runs failed, and why?"
```

With `-q` the chat asks nothing: only the answer goes to the standard output; tool lines, notices, the token line
and errors go to the standard error. It needs a service set up before (run the chat once without `-q`, or give
`--provider` and `--model`) and an accepted data notice: `--accept-data-notice` accepts it for good (it prints the
notice on the standard error). The exit code says what happened:

| Exit code | Meaning |
|---|---|
| 0 | answered |
| 1 | the AI service gave an error (the message says which) |
| 2 | something is not set up: no service, no model, no key, a chat without a terminal, or a `--tier` above the levels the model allows |
| 3 | the data notice for this service has not been accepted; nothing was sent |
| 130 | stopped with Ctrl-C |

The chat without `-q` needs a terminal: with piped input it stops (exit 2), so neither the data notice nor a key is
ever read from piped text.

### Commands

| Command | What it does |
|---|---|
| `/help` | The list of commands. |
| `/new` | Start a new conversation. |
| `/new carry` | Start a new conversation whose first question carries a short summary of this one. |
| `/retry` | Ask your last question again; its answer is replaced. |
| `/edit` | Change your last question and ask it again; its answer is replaced. On a terminal with GNU readline the question is put on the input line to change; elsewhere it is printed for you to copy, with `\` line ends. An empty line keeps the question and its answer. |
| `/model NAME` | Go on with another model of the same service; `/model` alone lists the models. |
| `/provider` | Choose another service. |
| `/privacy` | Where the data goes, the privacy settings sent with each request, and the settings only your account can change. |
| `/usage` | The tokens used today and this month for each service, then by month, service and model. |
| `/context` | What the assistant is told about your choices (project, input source, scenario) and, with a scenario, its model facts; after the first question also the model's context size, how much one tool result may hold and how often earlier results are shortened. |
| `/welcome` | The text on how the assistant works. |
| `/save` | Keep this service, model and level for the next start, and the free plan when `--free-tier` was given. |
| `/quit` | Leave the chat. |

To list the models of another service from the terminal, give its `--provider` ID (table below), for example
`flextool-assistant --list-models --provider ollama`. Add `--all` to list every model, also the ones the short list
leaves out.

### Services

| `--provider` | Service | Key variable | Sent with every request | Set in your account |
|---|---|---|---|---|
| `openai` | OpenAI | `OPENAI_API_KEY` | `store: false` (the requests are not kept for later viewing in your account) | data sharing and training settings of your organisation; zero data retention needs approval from OpenAI |
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` | — | data retention arrangements and training terms of your account |
| `mistral` | Mistral | `MISTRAL_API_KEY` | — | whether your plan allows training on your data, and the opt-out |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` | `provider.data_collection: deny` (only providers that do not collect your requests); optionally `provider.zdr: true` | your OpenRouter privacy settings, which apply on top |
| `gemini` | Google Gemini | `GEMINI_API_KEY` | — | free tier: Google may use prompts and answers to improve its products; paid tier terms differ |
| `ollama` | Ollama on this computer (`http://localhost:11434/v1`) | none | — | Ollama cloud models: check Ollama's terms yourself |
| `lmstudio` | LM Studio on this computer (`http://localhost:1234/v1`) | none | — | — |
| `openai_compatible` | any other service with the same request format as OpenAI (give `--base-url`) | none | — | check how the service handles data yourself |

The OpenRouter settings are in the settings file (see [Conversations and settings](#conversations-and-settings)):
`"openrouter.data_collection": "allow"` relaxes the first (the chat then says so), `"openrouter.zdr": true` uses
only providers that keep no copy of your requests, which narrows the list of models you can use, free models
especially.

FlexTool gives each service only the request headers it needs; the version, operating system and other
identification headers that the service libraries would add are not sent.

### Keys

The chat looks for the key of the service in this order and says where it found it:

1. the service's environment variable (table above, and [Environment variables](env_vars.md#ai-assistant));
2. the system keyring (Windows Credential Manager, macOS Keychain, the Secret Service on Linux);
3. the key file, only if you chose it.

`flextool-assistant --set-key --provider gemini` asks for the key without showing it and stores it in the system
keyring. Where there is no usable keyring, `--set-key --key-file` stores it unencrypted in `keys.json` in the
assistant's folder (on Linux and macOS readable by you only), after a warning. `flextool-assistant --forget-key --provider gemini`
removes a stored key after asking `[y/N]` (add `--yes` to skip the question in a script); **Forget key** in the settings asks too, with Cancel as the default. A key is never written to a project folder or a conversation, and never shown in a message.

### Models on this computer

**Ollama**: install Ollama and pull a model that supports tools (the model's page in the Ollama library lists
"tools"). Ollama loads models with a short context by default, too short for the assistant's instructions and
tools; FlexTool needs a context of at least 12 288 tokens, so start the Ollama server with, for example,
`OLLAMA_CONTEXT_LENGTH=16384` (or more). Before the first question the chat
loads the model and checks the context it runs with, and stops with that advice when it is too small. When the
model does not fit in the graphics card's memory, the chat says that it will be slower. FlexTool sends no limits of
threads or graphics memory: the Ollama server uses the computer as it is set up to. Ollama cloud models (names
ending in `:cloud` or `-cloud`, or models Ollama forwards to another host) run on Ollama's servers: the chat shows
them as such and asks for the data notice.

**LM Studio**: load a model in LM Studio with a context length of at least 12 288 tokens (for example 16 384) and start its server.
When the model is loaded, the chat checks its context before the first question.

Another server with the OpenAI request format: `--provider openai_compatible --base-url http://localhost:8000/v1`.
A server on this computer or your local network never goes through a proxy.

A model on this computer may take minutes to load or to read a long question, and a large model of a service may
think for minutes before it answers; the chat shows "Waiting for the model (45 s) — Ctrl-C stops" meanwhile. FlexTool
waits up to 15 minutes for a model on this computer or your local network and up to 5 minutes for a service on the
internet, counted from the last part of the answer that arrived.

### Levels

The start of the chat shows the level and why, for example "Level: basic (a model of about 14 billion
parameters)", followed by "Higher levels are not available for it." when the model allows no higher level. `--tier
basic` (or `standard`) chooses a lower level yourself; `/save` keeps the choice. A level above the ones the model
allows is lowered to the automatic level with a notice; with `-q` the chat stops instead (exit code 2) and sends
nothing. `--free-tier on` switches the free plan on for the session. See [Assistant levels](#assistant-levels).

### Conversations and settings

The chat keeps its files in the assistant folder of your user configuration: `~/.config/flextool/assistant/` on
Linux, `%LOCALAPPDATA%\flextool\assistant\` on Windows, `~/Library/Application Support/flextool/assistant/` on
macOS.

- **Conversations** are saved per project. `--continue` goes on with the newest conversation of the project,
  `--continue ID` with another one. A conversation open in another terminal or FlexTool window is not opened twice:
  the chat starts a new one. Conversations older than 30 days are deleted when the assistant starts.
- **Settings** (`settings.json`): the service, address, model and level that `/save` keeps, the privacy settings,
  `conversation_expiry_days` (default 30; `0` saves no conversations), `compaction_every` (how often older tool
  results are replaced by a short note so the conversation stays short) and `keep_recent_turns` (how many recent
  questions keep their tool results) — both automatic by default, from the model's context size (see
  [Budgets](#budgets)); a number overrides it — and `free_tier` (`null` automatic, `true` on, `false` off). Flags
  override the settings for one session.
- **Usage** (`usage.json`): the tokens of each month, service and model, and of each of the last 40 days, as each
  service reported them; FlexTool does not estimate what a service did not report, and shows a cost only when the
  service reports one. `flextool-assistant --usage` prints today's and this month's tokens of each service, then
  every month. It contains no questions or answers and is kept when conversations are
  deleted.
- **Acknowledgements** (`acknowledgements.json`): the data notices you accepted.

### Chat options

| Option | What it does |
|---|---|
| `--project-path FOLDER` / `--project NAME` | The project (as for the agent connection below). |
| `--provider ID` | The service for this session (table above). |
| `--model NAME` | The model for this session. `flextool-assistant --list-models --provider ID` lists the models of a service; `--test-model` checks that the model answers (one tiny request, no project data). |
| `--base-url URL` | The address of the service: a custom endpoint, or a model server on another port or computer. |
| `--tier basic\|standard\|advanced` | The assistant level: the automatic one or lower (see [Assistant levels](#assistant-levels)). |
| `--free-tier on\|off\|auto` | The free plan for this session (see [The free plan](#the-free-plan)); default: the saved setting, else automatic. |
| `--input-source SOURCE`, `--scenario NAME` | The input source and scenario the assistant uses when a question names none. |
| `--result-cap small\|medium\|large` | How much one tool result may hold (default: by level). |
| `-q QUESTION` | One question, then stop (see above). |
| `--accept-data-notice` | Accept the data notice of this service for good. |
| `--continue [ID]` | Go on with a saved conversation. |
| `--no-color` | No colours. |
| `--threads N` | Threads for FlexTool's own processing (default 1, so the assistant never slows down a running model). |
| `--verbose` | More messages on the standard error. |

## Connect an AI agent

An AI agent such as Claude Code can use the same tools with its own model. FlexTool starts them for the agent; the
agent's service gets everything the tools return (see [Where your data goes](#where-your-data-goes)).

### What an agent needs

- FlexTool installed (see [Install with FlexTool GUI](install_gui.md) or [Install with Spine Toolbox](install_toolbox.md)).
  The tools come with FlexTool; after an update, run "Update FlexTool" (or `pip install -e .` in a source checkout)
  so that the `flextool-assistant` command exists.
- An AI agent that can start tools on your computer through the Model Context Protocol (MCP), an open standard for
  connecting AI agents to tools. Claude Code is one; many other agents and editors support it too.
- A FlexTool project folder: the folder of a project of the FlexTool window (`projects/<project name>/`), or any
  folder with an `input_sources/` folder and the result folders FlexTool writes.

### Connect Claude Code

The tools only read, but what they return holds text from the project's databases and run logs: names,
descriptions and log lines that someone else may have written, for example in a database you were sent. FlexTool
tells the agent that tool results are data, not instructions, but an agent may still act on text written to look
like an instruction. If the agent can also run commands or write files on your computer, keep it asking you before
each such action.

Give Claude Code the command that starts the tools for one project. Use the absolute path of the project folder:

```bash
claude mcp add flextool -- flextool-assistant --mcp --project-path /home/me/flextool/projects/MyProject
```

Claude Code does not activate your Python environment, so if `claude mcp list` says `flextool-assistant` is not
found, give the full path of the command in the Python environment where FlexTool is installed, for example:

```bash
claude mcp add flextool -- /home/me/venv/bin/flextool-assistant --mcp --project-path /home/me/flextool/projects/MyProject
claude mcp add flextool -- C:\flextool\venv\Scripts\flextool-assistant.exe --mcp --project-path C:\flextool\projects\MyProject
```

`claude mcp add` connects the tools for the folder you run it in. Add `--scope user` (before `flextool`) to use them
in every folder. A Claude Code session that was already open sees a new connection only after it is started again.

With the Spine Toolbox bundle, start the tools with the bundle's own Python (the path depends on where you unpacked
the bundle):

```bash
claude mcp add flextool -- "C:\Spine Toolbox\_internal\Python\python.exe" -m flextool_assistant_entry --mcp --project-path C:\flextool\projects\MyProject
```

Then start Claude Code and ask, for example, "What is the existing capacity of coal_plant in scenario coal?" or
"Did the latest run of scenario base succeed, and if not, why?". Claude Code starts the tools itself when it needs
them; you do not start anything by hand. `claude mcp list` shows whether the connection works.

While the project is open in the FlexTool window, the agent's `get_gui_context` tool returns what is selected there
(the input sources, scenarios and results checked and highlighted, and what the result viewer shows), so you can
ask about "the selected run". The window keeps these names and selections in a small file in the assistant folder
(no paths, no data values, nothing from your chats) and deletes it when the project closes; the agent's service
gets them only when the agent calls the tool. Without the window, the tool says so and gives the options the tools
were started with. The tool also gives the model facts of the scenario it names (`model_facts`, as in "Ask"; read
by the connection itself, not from the window's file), and the connection asks the agent to call it first; a
connection with `--tier basic` has no such tool, so its agent gets no model facts.

To connect another project, add it under another name (`claude mcp add flextool-other -- …`), or remove the first
one (`claude mcp remove flextool`).

### Connect another agent

Agents that read a configuration file usually take the command and its arguments separately:

```json
{
  "mcpServers": {
    "flextool": {
      "command": "flextool-assistant",
      "args": ["--mcp", "--project-path", "/home/me/flextool/projects/MyProject"]
    }
  }
}
```

The tools talk to the agent through the command's standard input and output, so the agent must start the command
itself; there is no network port.

### Options

| Option | What it does |
|---|---|
| `--mcp` | Serve the tools to an AI agent. The agent starts the command with this option. |
| `--project-path FOLDER` | The project folder, as an absolute path. |
| `--project NAME` | The project `projects/NAME` under the folder the command starts in (the command prints the folder it uses). Use `--project-path` with agents: they may start the command in any folder. |
| `--input-source SOURCE` | The input source the tools use when a question names none: its file name or its number in the FlexTool window. Without it, the agent names a source when the project has several. |
| `--scenario NAME` | The scenario the tools use when a question names none. |
| `--result-cap small\|medium\|large` | How much one tool result may hold (rows, characters). Default: `medium`. A smaller cap saves the agent's context; the agent can still ask for more in one call, up to a fixed limit. |
| `--tier basic\|standard\|advanced` | The assistant level (see below). `basic` offers only the eight basic tools, in their shorter basic form, and starts with the small result cap; `standard` starts with the medium one and `advanced` with the large one. |
| `--threads N` | Threads for FlexTool's own processing in the tools. Default: 1, so the tools never slow down a model that is running. |
| `--verbose` | More messages on the standard error stream (the agent usually shows them in its log). |
| `--call TOOL [ARGUMENTS]` | Run one tool from the terminal and print its result (see below). |

### Levels for agents

The tools offer three prompts, one per assistant level. A prompt gives the agent instructions for working at that
level for the rest of the conversation:

- **basic** — look up and explain values with a few tool calls; for small models;
- **standard** — interpret results and compare scenarios;
- **advanced** — trace slack and penalties to their cause, read the model's equations, explain investment
  differences.

In Claude Code they are `/mcp__flextool__basic`, `/mcp__flextool__standard` and `/mcp__flextool__advanced`, and you
can add your question after the name. Without a prompt the agent works with the general instructions the tools
send when they connect. `--tier basic` limits the tools to the basic set, for agents that run a small model: any
other tool answers "The tool … is not part of the basic level. Restart the FlexTool connection without --tier basic
to use it." With `--tier basic` the tools are listed in the same shorter form the basic chat sends (`list_runs`, for
example, lists five of its arguments; the others still work when an agent sends them). Without `--tier`, and with
`--tier standard` or `--tier advanced`, every tool is listed in its full form, also beside the `basic` prompt: an
agent may pick any prompt, so it gets every tool. `--tier standard` and `--tier advanced` change only the default
result cap. The
tools do not know which model the agent uses, so they do not check the level against it: choose the level that suits
the agent's model (see [Assistant levels](#assistant-levels)).

## Assistant levels

The assistant works at one of three levels. The level changes what the model is asked to do, which tools it gets and
how many steps it may take; it never changes what a tool returns.

- **Basic** — look things up and explain them: one parameter, one result table or one documentation topic at a time,
  with eight tools (`search_docs`, `read_doc`, `get_entity`, `get_parameter`, `list_sources`, `list_runs`,
  `find_result_tables`, `query_result`) and at most six tool calls for each part of a question, three in one step
  (see [Budgets](#budgets)). A question
  that needs more, such as why there is slack or why two scenarios differ, gets what was found so far and the advice
  to use a higher level or a more capable model.
- **Standard** — interpret results and compare scenarios, linking a result to the inputs that drive it; every tool.
- **Advanced** — diagnose problems in several steps: trace slack and penalty costs to a limit and back to the inputs
  behind it, read the model's equations, and explain investment differences between scenarios; every tool.

**Which model for which question.** Smaller models and free plans suit look-up questions. For diagnosis — why there
is slack, why two scenarios differ — use a larger model or a paid plan. Free plans often limit how much text one
request may carry (for example a small number of input tokens per minute), and a diagnosis builds up long requests
as the assistant reads more results; such a request can then be refused however long you wait.

### How the level is chosen

FlexTool places each model from its name and, once it has checked the model, its context size (how much text the
model can read at once, counted in tokens, the pieces of text a service counts). The name rules are tried in order,
and the first that matches decides:

1. A small-model word in the name (`mini`, `nano`, `micro`, `tiny`, `lite`, `small`, `haiku`, `ministral`) → basic.
2. The size: the number of parameters a model server on this computer reports, else a size in the name such as
   `14b` or `8x7b`. Up to 34 billion → basic, up to 200 billion → standard, more → advanced. A count of only the
   active parameters (`a3b`) is not a size.
3. A fast or mid-range variant of a large family (`fast`, `maverick`, `scout`, `nova`, `sonar`, `deepseek-coder`)
   → standard.
4. The providers' largest families (`opus`, `sonnet`, `claude`, `large`, `max`, `grok`, `kimi`, `o3`, Gemini and GPT
   `pro`, DeepSeek `r1` and `v3`, and others) → advanced.
5. GPT with a major version of 5 or more → advanced.
6. Another known family (`flash`, `medium`, `gpt`, `gemini`, `mistral`, `codestral`, `command`, `glm`, and others)
   → standard.
7. A name FlexTool does not recognise: basic for a model on this computer or your local network, standard for a
   service on the internet. This is a guess.

The context size then sets the highest level:

| Context size (tokens) | Highest level |
|---|---|
| below 12 288 | none: no question is sent, and the message says how to give the model a larger context |
| 12 288 to 32 767 | basic |
| 32 768 to 65 535 | standard |
| 65 536 or more | advanced |

The automatic level is the lower of the two. You can choose a lower level, but not a higher one; for a model FlexTool
does not recognise, the higher levels are offered too, at your own risk, up to what the context size allows. A level
chosen before that is above the model's levels is lowered to the automatic level, with a notice. Until FlexTool has
checked the model (before its first question), the level comes from the name alone, and the settings say that it
may be lowered when FlexTool sees the model's context size.

The name rules are reviewed when a FlexTool version is prepared. If a model is placed at a level that does not suit
it, please report it.

### Budgets

Each level has safety caps on the work for one question. They keep a question from running on and keep the
requests within the model's context; they are not usage limits.

| | Basic | Standard | Advanced |
|---|---|---|---|
| Steps for each question (one request to the service each) | 3 | 8 | 12 |
| Tool calls for each question | 6 | 16 | 30 |
| Tool calls in one step | 3 | 8 | 8 |
| Longest answer (tokens; at most a quarter of the context size) | 4 096 | 8 000 | 16 000 |
| Continuations with a fresh context | 2 | 1 | none |
| Steps / tool calls / continuations with the free plan | 3 / 6 / 1 | 4 / 16 / none | 6 / 30 / none |

The model is asked to request only what the question needs, and to request lookups that do not depend on each
other together, in one step: one call with lists (for example several parameters of several entities with
`get_parameter`), or several calls in the same step. Each step is one request to the service, so this matters most
with the free plan, which counts steps; a step may hold the level's tool calls with the free plan too. With a
context size below 32 768 tokens a question part has only as many tool calls as their results fit the context
(for example, at 16 384 tokens: three, with results of up to 2 000 characters; six from about 22 000 tokens).

When a question reaches its limit of steps or tool calls and its level allows a continuation, the model writes a
short note of what it found so far and the next step (the note is not shown), and the question goes on with a fresh
context: the question, the note and the tool results the note needs, with new steps and tool calls. The fresh context
also lists the tool calls already made whose results it does not carry, each with what it returned, so the model
does not make them again. If a continued part still asks a tool for something the question already asked, the
assistant stops there: the model answers with what it has, and the chat ends with "The assistant kept trying the
same steps, so it stopped. A higher assistant level or a more capable model may get further." The chat shows
one line, for example "Continuing with a fresh context (part 2 of 3)…"; Stop works as at any other time. Every
request is counted in the usage. When no continuation is left, the model is asked to answer with what it has and to
say what is still unknown.

What each request carries of the earlier questions depends on the level, which keeps the requests of smaller models
short and focused. The saved conversation keeps everything.

| | Basic | Standard | Advanced |
|---|---|---|---|
| Earlier questions in each request | the last two, each with its final answer (text only) | every one, with its final answer (text only) | the whole conversation, with the tool calls and results |

The context block (your project, input source and scenario, and with a scenario its model facts) is always sent; the
model facts, too, go to the AI service: the names of the scenario's solves, periods, forecast branches and
timesets, its time steps, and the run folder name and state of its latest run with that run's error message. The context size also sets how much one
tool result may hold and, at the advanced level, how often the tool results of earlier questions are replaced by a
short note:

| Context size (tokens) | One tool result holds at most | Earlier results shortened |
|---|---|---|
| 12 288 to 32 767 | what fits the context (a few thousand characters) | at every question |
| 32 768 to 65 535 | the small result cap | every 4 questions, keeping the last 2 |
| 65 536 to 199 999 | the level's result cap, at most the medium one | every 6 questions, keeping the last 2 |
| 200 000 or more | the level's result cap (basic small, standard medium, advanced large) | every 6 questions, keeping the last 2 |

Before each request FlexTool estimates its size. A request that would not fit the context is first shortened (older
tool results are replaced by a note, and the chat says so); a request that still does not fit is not sent. A model on
this computer would otherwise silently drop the start of an over-long request.

### The free plan

Free plans limit how many requests you can send per minute and per day, and each step of a question is one request.
With the free plan on, each question uses fewer steps (see [Budgets](#budgets)), a tool result holds at most the
medium result cap, and the FlexTool window sends one question at a time. The first question says so: "Free plan:
the assistant uses at most 4 steps for each question to stay within the provider's request limits."

Automatically the free plan is on only for an OpenRouter model whose name ends `:free`. For any other service it is
off until you switch it on: in the window under Settings, "Free plan"; in the terminal with `--free-tier on` (`/save`
keeps it). When a service's limit on requests per minute or per day ends a question while the free plan is off,
the chat suggests switching it on, once for each service; nothing changes by itself.

### How fast requests are sent

One question may take several requests to the service: one for each round of tool calls, sent one after another.
That is why a limit on requests per second matters even when you ask one question at a time. FlexTool spaces the
requests to one service and model:

- When the service's answers say how many requests are left until a reset (OpenAI and Anthropic send this;
  Mistral sends the requests left in the current minute), FlexTool spreads the remaining requests over that time,
  and waits for the reset when none are left. This only ever makes the wait longer, never shorter than the waits
  below. A refusal whose answer says none are left waits for that reset before trying again.
- When the service refuses a request as too fast without saying how long to wait, FlexTool waits 1.1 seconds
  between requests from then on, doubling the wait after each further refusal (at most 20 seconds), and tries again
  by itself: up to 3 times, or 6 with the free plan on, all within 90 seconds. Each answered request shortens the
  wait again, down to 1.1 seconds. FlexTool remembers the wait for the next start. If the tries run out, the message
  says so, for example "Mistral limits how many requests per second a free plan may send. FlexTool now waits between
  requests; ask again."
- Mistral's requests are always at least 1.1 seconds apart, free plan or not (its free plan is limited per second;
  a paid account loses about a second per round of tool calls). With the free plan on, OpenRouter's free models
  start with 3 seconds (20 requests per minute). The learned wait never goes below these.
- A wait the service asks for is always kept.
- When the service says your account may send 0 requests per minute, nothing is tried again: the account's API
  access is not active yet. The message says so, for example "Your Mistral account allows 0 requests per minute:
  its API access is not active yet. Activate a plan in your Mistral account, then ask again." Test model in the
  settings shows the same message.

The chat shows the wait, for example "Mistral limits how fast requests may be sent. Sending the next one in 2
seconds."; Stop works during it. All chats of the FlexTool window that use the same service and model share one
wait.

## Units

Each node's `unit_of_measure` sets the units of its values
(see [Units of measure](reference/entities_and_time.md#units-of-measure)): a node without it is MWh, so its flows
are MW and its prices CUR/MWh; CUR is the model's currency. A node that sets
it (m3 for water) has its inputs and results in that unit (`inflow` in m3, `penalty_up` in CUR/m3, a flow in m3/h).
The tools show each unit as it is, say which node set it, and refuse to add up values of nodes of different units.
The assistant gives the unit the tools return; it raises a doubt only when a node's name or description contradicts
its unit. FlexTool labels units and does not convert them: if a model holds, say, kg in a node without
`unit_of_measure`, set it to `kg` so that the labels, and the assistant, say so.

## Try a tool from the terminal

`--call` runs one tool and prints its result as the agent would see it, which helps when something does not work as
expected:

```bash
flextool-assistant --project-path /home/me/flextool/projects/MyProject --call list_runs
flextool-assistant --project-path /home/me/flextool/projects/MyProject --call get_entity '{"name": ["coal_plant"], "scenario": "coal"}'
flextool-assistant --project-path /home/me/flextool/projects/MyProject --call get_parameter '{"entity_class": "unit", "parameters": ["existing", "invest_cost"], "entities": [["coal_plant"], ["wind_plant"]], "scenario": "coal"}'
```

The arguments are a JSON object, or `@file.json` to read them from a file. `get_parameter` takes lists of
parameters (up to 10) and entities (up to 20) and answers with one table: a row for each entity and parameter, with
the value, its shape and the alternative it comes from (or the default), and the units and notes once for each
parameter. A name it does not know is listed with the closest names, and the other rows are still answered. Its
`provenance` lists the scenario's alternatives highest rank first (`alternatives_highest_first`: the first one that
sets a value wins). `flextool-assistant --help` lists the options.

Other tools keep their results short so that a project with many runs fits the agent's context:

- `list_runs` lists the latest runs first: without filters the newest 8 (basic) or 20 (standard, advanced), with
  the number of runs in each state and, in `older`, the counts of the runs not shown; `"all": true` lists every
  run in the same order. A row is folder, state, `finished` (local time) and whether its results can be read. Runs
  with a run record come first, by when they finished (a run that has not finished, by when it started). A folder
  whose result files were written without a run record (by an older FlexTool) has the state `no_record`: its
  success is not known. Such runs come after the others, by when their files were last written. Copying a project
  changes those times, so the listing says that their order is uncertain; a page that lists none of them gives
  only their count. A result read from such a run says
  `"verified": false` (`compare_runs` lists such runs in `unverified`), and its run notes start with the same
  warning. Named folders (`"folders": ["base_1"]`), a filtered list of up to five runs or `"detail": "full"` add
  the verdict, times, FlexTool version and error; a value shared by every row is then given once
  (`same_in_every_row`). Filter with `source`, `scenario`, `state`, `results_usable` and `folders` (names or `*`
  patterns; a name or pattern that matches no folder is refused with the closest folder names); a filtered list
  holds every match, newest first. A long list comes in pages, and `shown` names the `offset` of the next one. The
  states are those of the run record (`succeeded`, `failed`, `killed`, `interrupted`, `crashed`, `running`,
  `abandoned`, `unknown`, `unreadable`, `malformed`) and `no_record`.
- `list_runs` with `"detail": "scan"` adds three numbers to each row: `total_cost` (the discounted cost over all
  periods, as the model minimises it, from `costs_discounted_p_`), `penalty_cost` (its slack penalty categories)
  and `largest_cost` (the category with the largest cost). Compare totals only between runs over the same periods.
  A run whose results may not be used gets no numbers, and `no_cost_table` names the runs without the table.
  `not_optimal_in` names the runs whose solve was not proven optimal (a time limit, a decomposition that did not
  converge), each with its verdict: their costs may be higher than the optimum. A
  given `detail` always wins: `"folders": ["a_1", "b_1"], "detail": "scan"` gives the costs of those two runs.
- `list_runs` with `"detail": "overview"` and one folder (or a `scenario` alone, for its newest run) gives one run in
  one call: its state and verdict, when it finished (local time, as in the rows), its costs by category (largest
  first), its capacity per unit, connection and
  storage (existing in the first period, invested added up over the periods, total in the last period; divested
  and capacity in construction when there is any) and the slacks it used, each with its largest entry. Capacity is
  never added up over periods or over entities. Entities with new, divested or in-construction capacity are listed
  first. When the overview is too long for one result, it drops, in this order: the cost categories under 1 % of
  the costs, entities beyond three per class, slacks beyond three, categories beyond three (of those of at least
  1 %), then fewer entities. A cut never brings back what an earlier one dropped, and each says what it shows. A
  request that selects no run says why: the folder is a run of another scenario, no run is of the scenario (with
  the closest scenario names), or the other filters leave nothing. A run whose results may not be used gets its state and error, and no numbers.
- `explain_entity` (standard and advanced levels) gives one unit, connection or node of one run in one call:
  `{"entity": "coal_plant", "run": "multi_year_1"}`, or a `scenario` alone for its newest run (a scenario whose
  latest run did not succeed is refused, as `query_result` refuses it). It lists the entity's inputs in the run's
  scenario as `get_entity` does (`invest_method` and `existing` also when no alternative sets them: `default` or
  `not set`), its flow and profile links with their parameters and its group, reserve and flowGroup memberships;
  then, per period and each with its table and unit: its capacity (existing, invested, divested, in construction,
  total; never added up over periods), its flows to and from each connected node (annualized; never added up over
  nodes), capacity factor, curtailment (VRE units), storage state, the prices at its nodes (mean, minimum and maximum
  over each period's time steps, unweighted), its non-zero discounted cost categories (their total over the periods
  first) and its investment value. A node gets its prices, slacks, costs and, for a storage, its capacity and
  state; the units and connections that feed it are not listed. `hints` say when a result is consistent with an
  input: what was invested equals `invest_max_total`, or `invest_max_period` in a period, a minimum or a
  retirement limit (each only when the `invest_method` uses that limit); `invest_method` allows no investment; or
  the investment value is positive (an upper limit holds the investment back) or negative in a period where
  nothing was invested (more capacity does not pay off). `not_in_run` names the tables the run has no values of
  the entity in, `too_large` those with too many values to summarise in one call (read them with `query_result`
  filtered to a period). When the input data changed after the run and no longer has the entity, its class is
  taken from the run's results. `entity_class` is needed only for a
  name that is a unit, a connection or a node at once. A long result is shortened in this order: the costs per
  period (totals kept), cost categories under 1 % of the entity's costs, maps summarised, the inputs that do not
  concern investment, the periods between the first and the last; then only the capacity and the hints are kept,
  then the capacity's total and fewer hints ("2 more hints") with only the note on what was left out, and last
  that note alone: the result always fits `max_chars`.
- `list_sources` with `"scenario": "coal_chp"` gives that scenario's alternatives, highest rank first, and its base
  alternative.
- `compare_runs` without `group_by`, `agg`, `sort` and `top_n` gives each run's totals per period and category
  (entities added up where that is meaningful; prices are kept per node), with the totals per period in `totals`;
  `"by_difference": true` gives each series, largest difference first. Energy and power are never added up over
  nodes, since the nodes of a model can hold different carriers (electricity, heat, hydrogen): such a table is
  totalled per node (units and connections added up into each node), and a table of units or connections without
  a node level keeps each unit or connection. Costs, emissions and other quantities that are not energy or power
  are added up over entities wherever a sum is meaningful. The first note says what was computed.
- `compare_inputs` counts the differences by alternative and entity class first, then gives one row per parameter
  and change with the entities it applies to (30 rows by default); `alternative` and `entity_class` narrow it. The
  `alternative` of a row is the one that makes the two scenarios differ (when each scenario's value comes from an
  alternative the other scenario does not have, `alternative` names b's, and the filter finds the row by either).
  Entities share a row only when their values are exactly the same.

## When something goes wrong

- **The agent says FlexTool could not start, and gives a command.** On some older processors FlexTool needs the
  compatible build of polars, its data library. The tools then start with only one tool, which gives the reason
  and the command that installs it. Run the command (or "Update FlexTool"), then restart the agent.
- **"The connection for AI assistants could not load the package mcp"** (in the agent's log). FlexTool was updated
  without its new requirements. Run "Update FlexTool", or `pip install -e .` in a source checkout.
- **The agent says a run's results are not used.** The tools read results only from a scenario whose latest run
  succeeded; files left in the folder by an earlier run are not used. The agent can read the run log to help get
  the scenario to run.
- **A spreadsheet input source is "not converted".** The tools read a spreadsheet through the database FlexTool
  makes from it when one of its scenarios runs. Run (or convert) a scenario of it first.
- **"No key found for …"** (chat). Set the service's environment variable, or store the key with
  `flextool-assistant --set-key --provider ID`.
- **"… is limiting how often requests can be sent"** or **"Your … quota is used up for now"** (chat). Free plans
  limit the requests per second, per minute and per day. The chat waits and tries again a few times by itself (see
  [How fast requests are sent](#how-fast-requests-are-sent)); after that, wait, or switch service or model with
  `/provider` or `/model`. `/usage` shows what was used. On a free plan, switch on
  the [free plan](#the-free-plan) so each question uses fewer requests. For Google, the line after the message
  names the quota that was hit (for example input tokens per minute, its limit and the model) and the wait Google
  suggests; the chat waits that long when it is short enough.
- **"… the daily quota of this model is used up; try tomorrow or another model"** (chat). Google's quota per day
  for the model is used up; it resets the next day (midnight Pacific time), so the chat does not wait for it.
- **"The level … is not available for this model, so nothing was sent."** (`-q`). The `--tier` asked for is above the
  levels the model allows; the message says why and which levels it allows.
- **"Nothing answers at localhost:11434. Start Ollama and try again."** Start the model server, then ask again.
- **"This model runs with a 2048-token context, too small …"** Restart Ollama with `OLLAMA_CONTEXT_LENGTH=16384`, or
  load the model in LM Studio with a larger context length.
- **"The model … does not support tool calls"**. Pick a model that supports tools; without them the assistant cannot
  read your data.
