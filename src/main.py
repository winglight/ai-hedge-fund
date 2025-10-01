import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.utils.visualize import save_graph_as_png
from src.cli.input import (
    parse_cli_inputs,
)
from src.data.providers import provider_context, DEFAULT_PROVIDER_NAME
from src.integrations.ibbot.strategy import build_strategy_bundle

import argparse
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import json

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
    data_provider: str = DEFAULT_PROVIDER_NAME,
    provider_options: dict | None = None,
    strategy_mode: bool = False,
):
    # Start progress tracking
    progress.start()

    try:
        # Build workflow (default to all analysts when none provided)
        workflow = create_workflow(selected_analysts if selected_analysts else None)
        agent = workflow.compile()

        with provider_context(data_provider, provider_options or {}):
            final_state = agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Make trading decisions based on the provided data.",
                        )
                    ],
                    "data": {
                        "tickers": tickers,
                        "portfolio": portfolio,
                        "start_date": start_date,
                        "end_date": end_date,
                        "analyst_signals": {},
                    },
                    "metadata": {
                        "show_reasoning": show_reasoning,
                        "model_name": model_name,
                        "model_provider": model_provider,
                        "ibbot_strategy_mode": strategy_mode,
                    },
                },
            )

        metadata = final_state.setdefault("metadata", {})
        metadata.setdefault("ibbot_strategy_mode", strategy_mode)

        final_message = final_state["messages"][-1] if final_state.get("messages") else None
        decisions = parse_hedge_fund_response(final_message.content) if final_message else {}
        metadata["parsed_decisions"] = decisions

        strategy_bundle = None
        strategy_error = None
        if strategy_mode:
            try:
                bundle = build_strategy_bundle(
                    decisions=decisions or {},
                    analyst_signals=final_state.get("data", {}).get("analyst_signals", {}),
                    provider=data_provider,
                    portfolio_agent=(getattr(final_message, "name", None) or "portfolio_manager"),
                    generated_at=datetime.now(timezone.utc),
                    context={
                        "model_name": model_name,
                        "model_provider": model_provider,
                    },
                )
                strategy_bundle = bundle.model_dump(mode="json", by_alias=True)
                metadata["ibbot_strategy_bundle"] = strategy_bundle
            except Exception as exc:  # pragma: no cover - defensive logging path
                strategy_error = str(exc)
                metadata["ibbot_conversion_error"] = strategy_error

        return {
            "decisions": decisions,
            "analyst_signals": final_state["data"]["analyst_signals"],
            "strategy_bundle": strategy_bundle,
            "strategy_error": strategy_error,
            "strategy_mode": strategy_mode,
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run the hedge fund trading system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=True,
        include_reasoning_flag=True,
    )

    tickers = inputs.tickers
    selected_analysts = inputs.selected_analysts

    # Construct portfolio here
    portfolio = {
        "cash": inputs.initial_cash,
        "margin_requirement": inputs.margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            }
            for ticker in tickers
        },
    }

    result = run_hedge_fund(
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        portfolio=portfolio,
        show_reasoning=inputs.show_reasoning,
        selected_analysts=inputs.selected_analysts,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
        data_provider=inputs.data_provider,
        provider_options=inputs.provider_options,
        strategy_mode=inputs.strategy_mode,
    )
    print_trading_output(result)
