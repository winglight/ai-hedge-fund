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
- [ibbot in Docker](#ibbot-in-docker)
- [How to Run](#how-to-run)
  - [⌨️ Command Line Interface](#️-command-line-interface)
  - [🖥️ Web Application (NEW!)](#️-web-application)
- [Contributing](#contributing)
- [Feature Requests](#feature-requests)
- [License](#license)

## How to Install

Before you can run the AI Hedge Fund, you'll need to install it and set up your API keys. These steps are common to both the full-stack web application and command line interface.

### 1. Clone the Repository

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. Set Up Your API Keys

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

If you plan to stream strategies into Interactive Brokers, continue with the [ibbot integration guide in the root README](../README.md#integrating-with-ibbot) before launching any containers.

**Important**: You must set at least one LLM API key (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`) for the hedge fund to work. 

**Financial Data**: Data for AAPL, GOOGL, MSFT, NVDA, and TSLA is free and does not require an API key. For any other ticker, you will need to set the `FINANCIAL_DATASETS_API_KEY` in the .env file.

## ibbot in Docker

Docker runs inherit credentials from the same root-level `.env` file the local workflow uses. To enable ibbot packaging inside containers:

1. Add the required variables described in the [Integrating with ibbot](../README.md#integrating-with-ibbot) section to your repository root `.env` file. The helper scripts automatically mount that file into `/app/.env` for every container.
2. If you keep credentials outside the repo, update the `volumes` entry in `docker-compose.yml` (and the helper scripts if needed) to mount your preferred secrets file into `/app/.env` before bringing the stack up.
3. For one-off overrides, export `IBBOT_HOST`, `IBBOT_ACCOUNT`, `IBBOT_ACCESS_TOKEN`, or `IBBOT_REFRESH_TOKEN` in your shell before launching the containers—Compose merges shell environment variables with values pulled from `.env`.

With credentials in place you can start an ibbot-enabled run directly from the Docker helper scripts:

```bash
cd docker
./run.sh --ticker AAPL \
         --data-provider ibbot \
         --strategy-mode intra_day \
         main
```

To stream the same session through Docker Compose (with Ollama) and observe server-sent events in the console, invoke Compose directly so you can pass the full argument list to Python:

```bash
docker compose run --rm hedge-fund-ollama \
  python src/main.py \
  --ticker AAPL \
  --data-provider ibbot \
  --strategy-mode intra_day \
  --stream
```

Run the command from the `docker/` directory so Compose picks up the bundled configuration files.

During execution you can monitor packaging progress with:

```bash
docker compose logs -f hedge-fund
# or, for Ollama-enabled runs
docker compose logs -f hedge-fund-ollama
```

Look for `ibbot_packaging_progress` lines to confirm the archive upload succeeded. When the job finishes the bundled artifact is written to `/app/results/<timestamp>/ibbot/` inside the running container. Use `docker cp <container>:/app/results/... .` or attach a volume that binds `/app/results` if you need the package on the host automatically.

## How to Run

### ⌨️ Command Line Interface

For users who prefer working with command line tools, you can run the AI Hedge Fund directly via terminal. This approach offers more granular control and is useful for automation, scripting, and integration purposes.

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

#### Using Docker

1. Make sure you have Docker installed on your system. If not, you can download it from [Docker's official website](https://www.docker.com/get-started).

2. Navigate to the docker directory:
```bash
cd docker
```

3. Build the Docker image:
```bash
# On Linux/Mac:
./run.sh build

# On Windows:
run.bat build
```

#### Running the AI Hedge Fund (with Docker)
```bash
# Navigate to the docker directory first
cd docker

# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA main
```

You can also specify a `--ollama` flag to run the AI hedge fund using local LLMs.

```bash
# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --ollama main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --ollama main
```

If you already have an Ollama server running (locally or on your network), point the containers at it instead of starting the bundled instance. You can either pass an explicit base URL or export `OLLAMA_BASE_URL` before running the scripts.

```bash
# Linux / macOS
./run.sh --ticker AAPL,MSFT,NVDA --ollama --ollama-base-url http://localhost:11434 main

# Windows
run.bat --ticker AAPL,MSFT,NVDA --ollama --ollama-base-url http://localhost:11434 main
```

When `OLLAMA_BASE_URL` is provided, the Docker compose services reuse that endpoint and the Ollama container is not started. To launch the embedded Ollama service manually with Docker Compose, add the `embedded-ollama` profile (e.g. `docker compose --profile embedded-ollama up`).

You can optionally specify the start and end dates to make decisions for a specific time period.

```bash
# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main
```

#### Running the Backtester (with Docker)
```bash
# Navigate to the docker directory first
cd docker

# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA backtest

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA backtest
```

**Example Output:**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />


You can optionally specify the start and end dates to backtest over a specific time period.

```bash
# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest
```

You can also specify a `--ollama` flag to run the backtester using local LLMs.
```bash
# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --ollama backtest

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --ollama backtest
```

## Contributing

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
