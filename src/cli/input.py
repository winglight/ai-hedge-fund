import os
import sys
import argparse
from datetime import datetime

from dateutil.relativedelta import relativedelta
import questionary
from colorama import Fore, Style

from src.llm.models import (
    LLM_ORDER,
    OLLAMA_LLM_ORDER,
    ModelProvider,
    get_model_info,
    get_provider_metadata,
)
from src.utils.analysts import ANALYST_ORDER
from src.utils.ollama import ensure_ollama_and_model
from src.data.providers import DEFAULT_PROVIDER_NAME, has_provider

from dataclasses import dataclass, field
from typing import Optional


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    require_tickers: bool = False,
    include_analyst_flags: bool = True,
    include_ollama: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--tickers",
        type=str,
        required=require_tickers,
        help="Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    if include_analyst_flags:
        parser.add_argument(
            "--analysts",
            type=str,
            required=False,
            help="Comma-separated list of analysts to use (e.g., michael_burry,other_analyst)",
        )
        parser.add_argument(
            "--analysts-all",
            action="store_true",
            help="Use all available analysts (overrides --analysts)",
        )
    if include_ollama:
        parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")
    return parser


def add_date_args(parser: argparse.ArgumentParser, *, default_months_back: int | None = None) -> argparse.ArgumentParser:
    if default_months_back is None:
        parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
        parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    else:
        parser.add_argument(
            "--end-date",
            type=str,
            default=datetime.now().strftime("%Y-%m-%d"),
            help="End date in YYYY-MM-DD format",
        )
        parser.add_argument(
            "--start-date",
            type=str,
            default=(datetime.now() - relativedelta(months=default_months_back)).strftime("%Y-%m-%d"),
            help="Start date in YYYY-MM-DD format",
        )
    return parser


def parse_tickers(tickers_arg: str | None) -> list[str]:
    if not tickers_arg:
        return []
    return [ticker.strip() for ticker in tickers_arg.split(",") if ticker.strip()]


def _get_first_env_value(*names: str) -> str | None:
    """Return the first non-empty environment variable from the provided names."""
    for name in names:
        value = os.getenv(name)
        if value:
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _parse_provider_from_env(raw_provider: str | None) -> ModelProvider | None:
    """Parse a provider string or enum name into a ModelProvider."""
    if not raw_provider:
        return None
    normalized = raw_provider.strip().lower()
    for provider in ModelProvider:
        if provider.value.lower() == normalized or provider.name.lower() == normalized:
            return provider
    return None


def resolve_model_from_env(use_ollama: bool) -> tuple[str | None, str | None]:
    """Resolve model selection from environment variables if provided."""
    provider_value = _get_first_env_value(
        "LLM_PROVIDER",
        "MODEL_PROVIDER",
        "AI_HEDGE_FUND_MODEL_PROVIDER",
    )
    model_value = _get_first_env_value(
        "LLM_MODEL",
        "MODEL_NAME",
        "AI_HEDGE_FUND_MODEL_NAME",
    )

    if use_ollama:
        ollama_model_value = _get_first_env_value("OLLAMA_MODEL") or model_value
        if ollama_model_value:
            return ollama_model_value, ModelProvider.OLLAMA.value

        provider_enum = _parse_provider_from_env(provider_value)
        if provider_enum == ModelProvider.OLLAMA and model_value:
            return model_value, provider_enum.value
        return None, None

    provider_enum = _parse_provider_from_env(provider_value)
    if provider_enum:
        if model_value:
            return model_value, provider_enum.value

        for _, name, provider in LLM_ORDER:
            if provider == provider_enum.value:
                return name, provider

    return None, None


def _announce_model_selection(model_name: str, model_provider: str) -> None:
    """Print a formatted summary of the chosen model and any capability hints."""
    model_info = get_model_info(model_name, model_provider)
    provider_metadata = get_provider_metadata()

    if model_info:
        provider_label = model_info.provider.value
        print(
            f"\nSelected {Fore.CYAN}{provider_label}{Style.RESET_ALL} model: "
            f"{Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}"
        )
    else:
        provider_label = model_provider
        print(
            f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}"
        )

    capabilities = model_info.get_capabilities() if model_info else provider_metadata.get(provider_label, {}).get("capabilities", {})

    capability_hints: list[str] = []
    if capabilities.get("supports_json_mode") is False:
        capability_hints.append(
            "JSON mode is not supported; outputs will be parsed heuristically."
        )
    capability_hints.extend(capabilities.get("notes", []))

    if capability_hints:
        print(f"{Fore.YELLOW}Capability hints:{Style.RESET_ALL}")
        for hint in capability_hints:
            print(f"  - {hint}")

    print()


def select_analysts(flags: dict | None = None) -> list[str]:
    if flags and flags.get("analysts_all"):
        return [a[1] for a in ANALYST_ORDER]

    if flags and flags.get("analysts"):
        return [a.strip() for a in flags["analysts"].split(",") if a.strip()]

    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done.",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)

    print(
        f"\nSelected analysts: {', '.join(Fore.GREEN + c.title().replace('_', ' ') + Style.RESET_ALL for c in choices)}\n"
    )
    return choices


def select_model(use_ollama: bool) -> tuple[str, str]:
    model_name: str = ""
    model_provider: str | None = None

    env_model, env_provider = resolve_model_from_env(use_ollama)
    if env_model and env_provider:
        _announce_model_selection(env_model, env_provider)
        return env_model, env_provider

    if use_ollama:
        print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")
        model_name = questionary.select(
            "Select your Ollama model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_name:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        if model_name == "-":
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
            sys.exit(1)

        model_provider = ModelProvider.OLLAMA.value
        _announce_model_selection(model_name, model_provider)
    else:
        model_choice = questionary.select(
            "Select your LLM model:",
            choices=[questionary.Choice(display, value=(name, provider)) for display, name, provider in LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        model_name, model_provider = model_choice

        model_info = get_model_info(model_name, model_provider)
        if model_info and model_info.is_custom():
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

        if not model_info:
            model_provider = model_provider or "Unknown"

        _announce_model_selection(model_name, model_provider)

    return model_name, model_provider or ""


def resolve_dates(start_date: str | None, end_date: str | None, *, default_months_back: int | None = None) -> tuple[str, str]:
    if start_date:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")
    if end_date:
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    final_end = end_date or datetime.now().strftime("%Y-%m-%d")
    if start_date:
        final_start = start_date
    else:
        months = default_months_back if default_months_back is not None else 3
        end_date_obj = datetime.strptime(final_end, "%Y-%m-%d")
        final_start = (end_date_obj - relativedelta(months=months)).strftime("%Y-%m-%d")
    return final_start, final_end


@dataclass
class CLIInputs:
    tickers: list[str]
    selected_analysts: list[str]
    model_name: str
    model_provider: str
    start_date: str
    end_date: str
    initial_cash: float
    margin_requirement: float
    show_reasoning: bool = False
    show_agent_graph: bool = False
    raw_args: Optional[argparse.Namespace] = None
    data_provider: str = DEFAULT_PROVIDER_NAME
    provider_options: dict[str, str] = field(default_factory=dict)
    strategy_mode: Optional[str] = None
    data_timeframe: Optional[str] = None
    stream: bool = False


def parse_cli_inputs(
    *,
    description: str,
    require_tickers: bool,
    default_months_back: int | None,
    include_graph_flag: bool = False,
    include_reasoning_flag: bool = False,
) -> CLIInputs:
    parser = argparse.ArgumentParser(description=description)

    # Common/interactive flags
    add_common_args(parser, require_tickers=require_tickers, include_analyst_flags=True, include_ollama=True)
    add_date_args(parser, default_months_back=default_months_back)

    # Funding flags (standardized, with alias)
    parser.add_argument(
        "--initial-cash",
        "--initial-capital",
        dest="initial_cash",
        type=float,
        default=100000.0,
        help="Initial cash position (alias: --initial-capital). Defaults to 100000.0",
    )
    parser.add_argument(
        "--margin-requirement",
        dest="margin_requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement ratio for shorts (e.g., 0.5 for 50%%). Defaults to 0.0",
    )

    if include_reasoning_flag:
        parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    if include_graph_flag:
        parser.add_argument("--show-agent-graph", action="store_true", help="Show the agent graph")

    parser.add_argument(
        "--data-provider",
        type=str,
        help="Market data provider to use (default: financial_datasets)",
    )
    parser.add_argument(
        "--provider-option",
        action="append",
        dest="provider_options",
        help="Provider-specific option in key=value format. Can be provided multiple times.",
    )
    parser.add_argument(
        "--strategy-mode",
        type=str,
        help="Trading strategy mode (e.g., swing, intra_day) used for IBBOT packaging.",
    )
    parser.add_argument(
        "--data-timeframe",
        type=str,
        help="Market data timeframe hint (e.g., 1d, 5m).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit structured Server-Sent Events to stdout for real-time monitoring.",
    )

    args = parser.parse_args()

    # Normalize parsed values
    tickers = parse_tickers(getattr(args, "tickers", None))
    selected_analysts = select_analysts({
        "analysts_all": getattr(args, "analysts_all", False),
        "analysts": getattr(args, "analysts", None),
    })
    model_name, model_provider = select_model(getattr(args, "ollama", False))
    start_date, end_date = resolve_dates(getattr(args, "start_date", None), getattr(args, "end_date", None), default_months_back=default_months_back)

    raw_provider = getattr(args, "data_provider", None) or os.getenv("AI_HEDGE_FUND_DATA_PROVIDER")
    if raw_provider:
        provider_value = raw_provider.lower()
        if not has_provider(provider_value):
            raise ValueError(f"Unknown data provider '{raw_provider}'")
    else:
        provider_value = DEFAULT_PROVIDER_NAME

    provider_options: dict[str, str] = {}
    for item in getattr(args, "provider_options", []) or []:
        if "=" not in item:
            raise ValueError(f"Provider option '{item}' must be in key=value format")
        key, value = item.split("=", 1)
        provider_options[key.strip()] = value.strip()

    return CLIInputs(
        tickers=tickers,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
        start_date=start_date,
        end_date=end_date,
        initial_cash=getattr(args, "initial_cash", 100000.0),
        margin_requirement=getattr(args, "margin_requirement", 0.0),
        show_reasoning=getattr(args, "show_reasoning", False),
        show_agent_graph=getattr(args, "show_agent_graph", False),
        raw_args=args,
        data_provider=provider_value,
        provider_options=provider_options,
        strategy_mode=getattr(args, "strategy_mode", None),
        data_timeframe=getattr(args, "data_timeframe", None),
        stream=getattr(args, "stream", False),
    )


