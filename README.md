# AI Hedge Fund

This is a proof of concept for an AI-powered hedge fund.  The goal of this project is to explore the use of AI to make trading decisions.  This project is for **educational** purposes only and is not intended for real trading or investment.

This system employs several agents working together:

1. Aswath Damodaran Agent - The Dean of Valuation, focuses on story, numbers, and disciplined valuation
2. Ben Graham Agent - The godfather of value investing, only buys hidden gems with a margin of safety
3. Bill Ackman Agent - An activist investor, takes bold positions and pushes for change
4. Cathie Wood Agent - The queen of growth investing, believes in the power of innovation and disruption
5. Charlie Munger Agent - Warren Buffett's partner, only buys wonderful businesses at fair prices
6. Michael Burry Agent - The Big Short contrarian who hunts for deep value
7. Mohnish Pabrai Agent - The Dhandho investor, who looks for doubles at low risk
8. Peter Lynch Agent - Practical investor who seeks "ten-baggers" in everyday businesses
9. Phil Fisher Agent - Meticulous growth investor who uses deep "scuttlebutt" research 
10. Rakesh Jhunjhunwala Agent - The Big Bull of India
11. Stanley Druckenmiller Agent - Macro legend who hunts for asymmetric opportunities with growth potential
12. Warren Buffett Agent - The oracle of Omaha, seeks wonderful companies at a fair price
13. Valuation Agent - Calculates the intrinsic value of a stock and generates trading signals
14. Sentiment Agent - Analyzes market sentiment and generates trading signals
15. Fundamentals Agent - Analyzes fundamental data and generates trading signals
16. Technicals Agent - Analyzes technical indicators and generates trading signals
17. Risk Manager - Calculates risk metrics and sets position limits
18. Portfolio Manager - Makes final trading decisions and generates orders

<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

Note: the system does not actually make any trades.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [How to Install](#how-to-install)
- [Integrating with ibbot](#integrating-with-ibbot)
- [How to Run](#how-to-run)
  - [⌨️ Command Line Interface](#️-command-line-interface)
  - [🖥️ Web Application](#️-web-application)
- [How to Contribute](#how-to-contribute)
- [Feature Requests](#feature-requests)
- [License](#license)

## How to Install

Before you can run the AI Hedge Fund, you'll need to install it and set up your API keys. These steps are common to both the full-stack web application and command line interface.

### 1. Clone the Repository

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. Set up API keys

Create a `.env` file for your API keys:
```bash
# Create .env file for your API keys (in the root directory)
cp .env.example .env
```

Open and edit the `.env` file to add your API keys:
```bash
# For running LLMs hosted by openai (gpt-4o, gpt-4o-mini, etc.)
OPENAI_API_KEY=your-openai-api-key

# For getting financial data to power the hedge fund
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

**Important**: You must set at least one LLM API key (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`) for the hedge fund to work. 

**Financial Data**: Data for AAPL, GOOGL, MSFT, NVDA, and TSLA is free and does not require an API key. For any other ticker, you will need to set the `FINANCIAL_DATASETS_API_KEY` in the .env file.

## Integrating with ibbot

Interactive Brokers Bot (ibbot) packaging adds an execution-ready hand-off to every run. Before enabling it, populate the required credentials in your `.env` file (or the environment in which you launch the app):

- `IBBOT_HOST`
- `IBBOT_ACCOUNT`
- `IBBOT_ACCESS_TOKEN`
- `IBBOT_REFRESH_TOKEN`

These values allow the platform to authenticate with ibbot and upload the packaged strategy bundle. If you're running inside Docker, follow the [container-specific setup instructions](docker/README.md#ibbot-in-docker) first so the compose services receive the credentials.

To direct the system to use ibbot pricing and generate an intraday package:

- **CLI:** pass both `--data-provider ibbot` and `--strategy-mode intra_day` along with your normal arguments. The data provider flag tells the loader to fetch prices from ibbot, while the strategy mode ensures the risk and portfolio agents produce a bundle that matches ibbot's intraday constraints.
- **Web UI:** choose **ibbot** from the *Data Provider* menu on the workflow start node, then pick **Intra-day** from the *Strategy* dropdown. The live run drawer shows an "ibbot packaging" progress step and updates to "Ready to export" once the bundle is uploaded. If packaging fails, the error message surfaces next to the provider badge.

When both flags are enabled the results panel displays the decision JSON and a downloadable `ibbot_bundle.zip` (or similar archive) so you can immediately import the strategy inside Interactive Brokers.

## How to Run

### ⌨️ Command Line Interface

You can run the AI Hedge Fund directly via terminal. This approach offers more granular control and is useful for automation, scripting, and integration purposes.

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

#### Quick Start

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

#### Run the AI Hedge Fund
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

You can also specify a `--ollama` flag to run the AI hedge fund using local LLMs.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

You can optionally specify the start and end dates to make decisions over a specific time period.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

#### Worked example: Streaming an ibbot-enabled session

Once your `.env` includes the ibbot credentials, you can stream an intraday run end-to-end. The example below requests SSE updates so you can watch analysts and packaging progress in real time:

```bash
poetry run python src/main.py \
  --ticker AAPL \
  --data-provider ibbot \
  --strategy-mode intra_day \
  --stream
```

During execution the terminal prints server-sent events such as `event: analyst_update`, `event: risk_update`, and `event: ibbot_packaging_progress` so you can track each phase. When the session completes the CLI reports the location of the packaged archive inside `results/<timestamp>/ibbot/`. The folder contains the serialized bundle (`ibbot_bundle.zip`) alongside the traditional decision log so you can reference both artifacts later.

#### Enable IBBOT strategy mode

Packaging the run output for Interactive Brokers Bots (IBBOT) requires enabling strategy mode so the risk and portfolio agents
optimize trades for the appropriate horizon.

- **CLI:** pass `--strategy-mode` (for example `--strategy-mode intra_day` or `--strategy-mode swing`) to activate packaging. You
  can optionally pair it with `--data-timeframe` to hint the market data cadence sent to analysts.
- **Web app:** choose a strategy from the *Strategy* dropdown on the portfolio start node. The selection is saved with the flow
  so subsequent runs reuse it automatically.
- **Persisted workflows:** strategy mode and timeframe are stored in `workflow_metadata` and propagate through the graph so the
  risk and portfolio agents align position limits with the requested style.

When strategy mode is enabled the final run output includes an IBBOT-compatible bundle alongside the legacy decision JSON. If
packaging fails, the UI surfaces a clear conversion error while still displaying analyst decisions.

#### Run the Backtester
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**Example Output:**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />


Note: The `--ollama`, `--start-date`, and `--end-date` flags work for the backtester, as well!

### 🖥️ Web Application

The new way to run the AI Hedge Fund is through our web application that provides a user-friendly interface. This is recommended for users who prefer visual interfaces over command line tools.

Please see detailed instructions on how to install and run the web application [here](https://github.com/virattt/ai-hedge-fund/tree/main/app).

To mirror the CLI flow above, start a new workflow, select **ibbot** as the data provider, and choose **Intra-day** strategy mode. Hit *Run* to watch the right-hand activity stream surface the same SSE events (analyst updates, packaging status) that the CLI prints. When packaging finishes a new *ibbot Bundle* tile appears in the run results drawer; click it to download the archive. If you're deploying via containers, remember to review the [Docker instructions](docker/README.md#ibbot-in-docker) for mounting credentials before launching the web stack.

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />


## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
