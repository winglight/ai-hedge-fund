import logging
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.integrations.ibbot.strategy import (
    StrategyConversionError,
    attach_strategy_bundle,
)
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.utils.visualize import save_graph_as_png
from src.cli.input import (
    parse_cli_inputs,
)
from src.data.providers import provider_context, DEFAULT_PROVIDER_NAME

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)

logger = logging.getLogger(__name__)


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
    strategy_mode: str | None = None,
    data_timeframe: str | None = None,
):
    # Start progress tracking
    progress.start()

    try:
        # Build workflow (default to all analysts when none provided)
        workflow = create_workflow(selected_analysts if selected_analysts else None)
        agent = workflow.compile()

        with provider_context(data_provider, provider_options or {}):
            data_granularity = (
                "intraday"
                if (strategy_mode and "intra" in strategy_mode.lower())
                or (data_timeframe and any(ch in data_timeframe.lower() for ch in ["m", "min", "hour", "hr", "h"]))
                else "end_of_day"
            )
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
                        "workflow_metadata": {
                            "strategy_mode": strategy_mode,
                            "data_timeframe": data_timeframe,
                            "data_provider": data_provider,
                            "data_granularity": data_granularity,
                        },
                    },
                    "metadata": {
                        "show_reasoning": show_reasoning,
                        "model_name": model_name,
                        "model_provider": model_provider,
                        "strategy_mode": strategy_mode,
                        "data_timeframe": data_timeframe,
                        "data_granularity": data_granularity,
                    },
                },
            )

        decisions = parse_hedge_fund_response(final_state["messages"][-1].content)
        final_state.setdefault("data", {})
        if decisions is not None:
            final_state["data"]["portfolio_decisions"] = decisions

        if decisions:
            try:
                attach_strategy_bundle(final_state, decisions)
            except StrategyConversionError as exc:
                logger.warning("IBBOT strategy packaging failed: %s", exc)
            except Exception:
                logger.exception("Unexpected error while preparing IBBOT strategy bundle")
                final_state["data"]["ibbot_strategy"] = {
                    "available": False,
                    "error": "unexpected_error",
                }
        else:
            final_state["data"]["ibbot_strategy"] = {
                "available": False,
                "error": "missing_decisions",
            }

        workflow_metadata = final_state["data"].get("workflow_metadata", {})
        strategy_mode = final_state["metadata"].get("strategy_mode") or workflow_metadata.get("strategy_mode")
        data_timeframe = final_state["metadata"].get("data_timeframe") or workflow_metadata.get("data_timeframe")

        return {
            "decisions": decisions,
            "analyst_signals": final_state["data"].get("analyst_signals", {}),
            "current_prices": final_state["data"].get("current_prices", {}),
            "ibbot_strategy": final_state["data"].get("ibbot_strategy", {"available": False}),
            "strategy_mode": strategy_mode,
            "data_timeframe": data_timeframe,
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


def _stream_handler_factory():
    """Return a handler that prints progress updates as SSE events."""

    def handler(agent_name, ticker, status, analysis, timestamp, context):
        event = "analyst_update"
        normalized = agent_name or ""
        if normalized.startswith("risk_management"):
            event = "risk_update"
        elif normalized.startswith("portfolio_manager"):
            event = "portfolio_update"

        payload = {
            "agent": normalized,
            "ticker": ticker,
            "status": status,
            "analysis": analysis,
            "timestamp": timestamp,
            "context": context or {},
        }

        serialized = json.dumps({k: v for k, v in payload.items() if v is not None})
        print(f"event: {event}")
        print(f"data: {serialized}")
        print()
        sys.stdout.flush()

    return handler


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

    stream_handler = None
    if inputs.stream:
        progress.set_live_enabled(False)
        stream_handler = progress.register_handler(_stream_handler_factory())

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

    try:
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
            data_timeframe=inputs.data_timeframe,
        )
    finally:
        if stream_handler:
            progress.unregister_handler(stream_handler)
            progress.set_live_enabled(True)

    print_trading_output(result)
