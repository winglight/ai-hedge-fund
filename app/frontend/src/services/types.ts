// Shared types for API requests and responses
export enum ModelProvider {
  ALIBABA = 'Alibaba',
  ANTHROPIC = 'Anthropic',
  AZURE_OPENAI = 'Azure OpenAI',
  DEEPSEEK = 'DeepSeek',
  GEMINI = 'Gemini',
  GOOGLE = 'Google',
  GIGACHAT = 'GigaChat',
  GROQ = 'Groq',
  META = 'Meta',
  MISTRAL = 'Mistral',
  OLLAMA = 'Ollama',
  OPENAI = 'OpenAI',
  OPENROUTER = 'OpenRouter',
  XAI = 'xAI',
}

export interface ProviderCapabilities {
  supports_json_mode?: boolean;
  supports_reasoning?: boolean;
  notes?: string[];
  api_key_env?: string[];
}

export interface AgentModelConfig {
  agent_id: string;
  model_name?: string;
  model_provider?: ModelProvider;
}

export interface GraphNode {
  id: string;
  type?: string;
  data?: any;
  position?: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type?: string;
  data?: any;
}

export interface PortfolioPosition {
  ticker: string;
  quantity: number;
  trade_price: number;
}

// Base interface for shared fields between HedgeFundRequest and BacktestRequest
export interface BaseHedgeFundRequest {
  tickers: string[];
  graph_nodes: GraphNode[];
  graph_edges: GraphEdge[];
  agent_models?: AgentModelConfig[];
  model_name?: string;
  model_provider?: ModelProvider;
  margin_requirement?: number;
  portfolio_positions?: PortfolioPosition[];
  api_keys?: Record<string, string>;
  data_provider?: string;
  data_provider_options?: Record<string, string>;
  strategy_mode?: string;
  data_timeframe?: string;
}

export interface HedgeFundRequest extends BaseHedgeFundRequest {
  end_date?: string;
  start_date?: string;
  initial_cash?: number;
}

export interface BacktestRequest extends BaseHedgeFundRequest {
  start_date: string;
  end_date: string;
  initial_capital?: number;
}

export interface BacktestDayResult {
  date: string;
  portfolio_value: number;
  cash: number;
  decisions: Record<string, any>;
  executed_trades: Record<string, number>;
  analyst_signals: Record<string, any>;
  current_prices: Record<string, number>;
  long_exposure: number;
  short_exposure: number;
  gross_exposure: number;
  net_exposure: number;
  long_short_ratio: number | null;
}

export interface BacktestPerformanceMetrics {
  sharpe_ratio?: number;
  sortino_ratio?: number;
  max_drawdown?: number;
  max_drawdown_date?: string;
  long_short_ratio?: number;
  gross_exposure?: number;
  net_exposure?: number;
}

export interface StrategySignalPayload {
  symbol: string;
  action: string;
  quantity: number;
  confidence?: number | null;
  rationale?: string | null;
  source_agent: string;
  model_provider?: string | null;
  generated_at: string;
  metadata?: Record<string, any>;
}

export interface RiskDirectivePayload {
  symbol: string;
  max_notional?: number | null;
  max_shares?: number | null;
  reference_price?: number | null;
  source_agent: string;
  generated_at: string;
  metadata?: Record<string, any>;
}

export interface IbbotStrategyBundle {
  data_provider?: string | null;
  model_provider?: string | null;
  generated_at: string;
  strategy_mode?: string | null;
  data_timeframe?: string | null;
  workflow_metadata?: Record<string, any>;
  raw_decisions: Record<string, any>;
  signals: StrategySignalPayload[];
  risk_directives: RiskDirectivePayload[];
}

export interface StrategyPackagingStatus {
  available: boolean;
  error?: string;
  bundle?: IbbotStrategyBundle;
}