"""
Ad Optimization Agent — Starter Code (Python, LangGraph/LangChain optional)

What this gives you
-------------------
• A runnable, simulation-first ad optimization agent that can later plug into real ad APIs.
• Multi-armed bandit (Thompson Sampling) for budget/bid allocation across channels/creatives.
• Agent graph (LangGraph-style) with nodes: fetch_metrics → analyze → propose_actions → (optional) human_approve → execute → log.
• Simple policy + transparent logs + responsible AI guardrails (budget caps, blocklists, explainability record).

How to run (simulation)
-----------------------
python ad_agent_starter.py  # runs a short training loop with a mock ad platform

To integrate with LangGraph
---------------------------
• pip install langgraph langchain
• Switch USE_LANGGRAPH=True below to build/run as a graph; else, a simple Python loop runs.

Notes
-----
• Replace MockAdPlatform with your connectors (e.g., Google Ads, Meta, DV360, LinkedIn). Keep the same interface.
• This is starter code—trim or expand nodes as you need. Heavier orchestration (e.g., multi-agent) can be added via parallel graphs.
"""
from __future__ import annotations
import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

# Toggle these to adopt LangGraph/LangChain orchestration.
USE_LANGGRAPH = False

try:
    if USE_LANGGRAPH:
        from langgraph.graph import StateGraph, END
except Exception:
    USE_LANGGRAPH = False

############################
# Domain models & utilities
############################

Channel = str   # e.g., "search", "social", "display"
Creative = str  # e.g., creative ids or names

@dataclass
class Arm:
    channel: Channel
    creative: Creative

    def key(self) -> str:
        return f"{self.channel}::{self.creative}"

@dataclass
class Metrics:
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    cost: float = 0.0

@dataclass
class Action:
    arm_key: str
    bid_multiplier: float  # e.g., 0.8 - 1.2
    budget_delta: float    # +/- amount in currency
    rationale: str

@dataclass
class PolicyConfig:
    min_bid_mult: float = 0.80
    max_bid_mult: float = 1.20
    max_budget_delta_pct: float = 0.20  # cap change vs. current budget
    daily_budget_cap: float = 10_000.0
    guardrail_blocklist: List[str] = field(default_factory=list)  # keywords/segments
    require_human_approval: bool = False

@dataclass
class AgentState:
    # Current belief about arms (posterior for CTR/CVR via Beta distributions)
    alpha_click: Dict[str, float] = field(default_factory=dict)
    beta_click: Dict[str, float] = field(default_factory=dict)
    alpha_conv: Dict[str, float] = field(default_factory=dict)
    beta_conv: Dict[str, float] = field(default_factory=dict)

    # Running budgets
    budgets: Dict[str, float] = field(default_factory=dict)

    # Latest observed metrics
    latest_metrics: Dict[str, Metrics] = field(default_factory=dict)

    # Proposed & executed actions
    proposed_actions: List[Action] = field(default_factory=list)
    executed_actions: List[Action] = field(default_factory=list)

    # Responsible AI logs
    explainer_log: List[Dict] = field(default_factory=list)

    # Misc
    step: int = 0

############################
# Mock Ad Platform (simulated)
############################

class MockAdPlatform:
    """A fake ad platform that simulates outcomes.
    Replace with real API client (Google Ads, Meta, etc.).
    """
    def __init__(self, arms: List[Arm], seed: int = 7):
        self.rng = random.Random(seed)
        self.true_ctr: Dict[str, float] = {}
        self.true_cvr: Dict[str, float] = {}
        self.ecpm_baseline: Dict[str, float] = {}
        self.budgets: Dict[str, float] = {}
        self.bids: Dict[str, float] = {}

        for arm in arms:
            k = arm.key()
            # Ground-truth performance (unknown to agent)
            self.true_ctr[k] = self.rng.uniform(0.005, 0.04)
            self.true_cvr[k] = self.rng.uniform(0.01, 0.12)
            self.ecpm_baseline[k] = self.rng.uniform(1.5, 6.0)  # $ eCPM baseline
            self.budgets[k] = 200.0
            self.bids[k] = 1.0

    def get_metrics(self) -> Dict[str, Metrics]:
        # Simulate one time-slice of delivery given budgets and bids
        out: Dict[str, Metrics] = {}
        for k in self.budgets:
            budget = self.budgets[k]
            bid = self.bids[k]
            # Impressions scale with budget & bid
            imps = int(budget * bid * self.ecpm_baseline[k] * 50)
            ctr = self.true_ctr[k] * (0.8 + 0.4 * self.rng.random())
            cvr = self.true_cvr[k] * (0.8 + 0.4 * self.rng.random())
            clicks = int(imps * ctr)
            convs = int(max(0, clicks) * cvr)
            cost = budget  # simple: spend ~budget per slice
            out[k] = Metrics(impressions=imps, clicks=clicks, conversions=convs, cost=cost)
        return out

    def update_bids(self, changes: Dict[str, float]):
        for k, mult in changes.items():
            self.bids[k] = max(0.5, min(2.0, self.bids.get(k, 1.0) * mult))

    def update_budgets(self, changes: Dict[str, float]):
        for k, delta in changes.items():
            self.budgets[k] = max(0.0, self.budgets.get(k, 0.0) + delta)

############################
# Policy / Learning (Bandit)
############################

class ThompsonBandit:
    """Thompson Sampling for clicks and conversions jointly.
    • We maintain Beta posteriors for CTR and CVR.
    • Use sampled CTR/CVR to compute expected convs/$$ and allocate.
    """
    def __init__(self, state: AgentState):
        self.s = state

    def ensure_arm(self, k: str):
        self.s.alpha_click.setdefault(k, 1.0)
        self.s.beta_click.setdefault(k, 1.0)
        self.s.alpha_conv.setdefault(k, 1.0)
        self.s.beta_conv.setdefault(k, 1.0)
        self.s.budgets.setdefault(k, 200.0)

    def update_posteriors(self, metrics: Dict[str, Metrics]):
        for k, m in metrics.items():
            self.ensure_arm(k)
            # Click model: Beta(α_click + clicks, β_click + imps - clicks)
            self.s.alpha_click[k] += m.clicks
            self.s.beta_click[k] += max(0, m.impressions - m.clicks)
            # Conversion model: Beta(α_conv + conv, β_conv + clicks - conv)
            self.s.alpha_conv[k] += m.conversions
            self.s.beta_conv[k] += max(0, m.clicks - m.conversions)

    def sample_value(self, k: str, rng: random.Random) -> float:
        # Lazy Beta sampler without numpy/scipy
        def beta_sample(a: float, b: float) -> float:
            # Use simple approximation via inverse transform of two Gamma draws
            # For starter code: fallback using random.gammavariate
            x = rng.gammavariate(a, 1.0)
            y = rng.gammavariate(b, 1.0)
            return x / (x + y) if (x + y) > 0 else 0.0
        ctr = beta_sample(self.s.alpha_click[k], self.s.beta_click[k])
        cvr = beta_sample(self.s.alpha_conv[k], self.s.beta_conv[k])
        # Proxy value: expected conversions per 1k imps, or ROI proxy
        return ctr * cvr

    def recommend(self, rng: random.Random, top_n: int = 3) -> List[Tuple[str, float]]:
        samples = []
        for k in self.s.budgets.keys():
            v = self.sample_value(k, rng)
            samples.append((k, v))
        samples.sort(key=lambda kv: kv[1], reverse=True)
        return samples[:top_n]

############################
# Nodes (Agent graph stages)
############################

def fetch_metrics_node(state: AgentState, platform: MockAdPlatform, policy: ThompsonBandit) -> AgentState:
    metrics = platform.get_metrics()
    state.latest_metrics = metrics
    policy.update_posteriors(metrics)
    state.step += 1
    state.explainer_log.append({
        "step": state.step,
        "event": "fetch_metrics",
        "observations": {k: asdict(v) for k, v in metrics.items()}
    })
    return state

def analyze_node(state: AgentState, rng: random.Random, policy: ThompsonBandit) -> AgentState:
    # Rank arms by sampled conversion value
    ranked = policy.recommend(rng, top_n=len(state.budgets))
    state.explainer_log.append({
        "step": state.step,
        "event": "analyze",
        "ranked_arms": ranked[:5]
    })
    return state

def propose_actions_node(state: AgentState, cfg: PolicyConfig) -> AgentState:
    # Simple rule: boost top 30%, reduce bottom 30%
    n = len(state.budgets)
    if n == 0:
        return state
    sorted_arms = sorted(state.budgets.keys(), key=lambda k: state.alpha_conv.get(k,1)/max(1,state.beta_conv.get(k,1)), reverse=True)
    top_cut = max(1, n // 3)
    bottom_cut = max(1, n // 3)
    top_arms = set(sorted_arms[:top_cut])
    bottom_arms = set(sorted_arms[-bottom_cut:])

    actions: List[Action] = []
    for k in state.budgets.keys():
        cur_budget = state.budgets[k]
        if k in top_arms:
            bid_mult = min(cfg.max_bid_mult, 1.05)
            budget_delta = min(cur_budget * cfg.max_budget_delta_pct, 50.0)
            rationale = "Top performer — modest boost to bid and budget"
        elif k in bottom_arms:
            bid_mult = max(cfg.min_bid_mult, 0.95)
            budget_delta = -min(cur_budget * cfg.max_budget_delta_pct, 50.0)
            rationale = "Underperformer — cautious downshift to reallocate spend"
        else:
            bid_mult = 1.0
            budget_delta = 0.0
            rationale = "Stable — no change"
        # Guardrails: skip if any blocklist match (toy example)
        if any(term.lower() in k.lower() for term in cfg.guardrail_blocklist):
            rationale += " (SKIPPED: guardrail blocklist)"
            bid_mult = 1.0
            budget_delta = 0.0
        actions.append(Action(arm_key=k, bid_multiplier=bid_mult, budget_delta=budget_delta, rationale=rationale))

    # Enforce daily cap across all budgets after deltas (soft check)
    projected_total = sum(state.budgets.values()) + sum(a.budget_delta for a in actions)
    if projected_total > cfg.daily_budget_cap:
        # Scale down uniformly
        scale = cfg.daily_budget_cap / max(1e-9, projected_total)
        for a in actions:
            a.budget_delta *= scale
        state.explainer_log.append({
            "step": state.step,
            "event": "budget_cap",
            "projected_total": projected_total,
            "cap": cfg.daily_budget_cap,
            "scaled_by": scale,
        })

    state.proposed_actions = actions
    state.explainer_log.append({
        "step": state.step,
        "event": "propose_actions",
        "actions": [asdict(a) for a in actions]
    })
    return state

def human_approve_node(state: AgentState, cfg: PolicyConfig) -> AgentState:
    if not cfg.require_human_approval:
        state.explainer_log.append({"step": state.step, "event": "auto_approved"})
        return state
    # Placeholder: In production, route actions for human review (UI, Slack, etc.)
    # For now, we simulate approval of non-zero deltas only if rationale contains "boost"
    approved: List[Action] = []
    for a in state.proposed_actions:
        ok = ("boost" in a.rationale) or (a.bid_multiplier == 1.0 and a.budget_delta == 0.0)
        if not ok:
            a.budget_delta = 0.0
            a.bid_multiplier = 1.0
            a.rationale += " (HUMAN OVERRIDE: neutralized)"
        approved.append(a)
    state.proposed_actions = approved
    state.explainer_log.append({"step": state.step, "event": "human_approved", "count": len(approved)})
    return state

def execute_actions_node(state: AgentState, platform: MockAdPlatform) -> AgentState:
    bid_changes = {a.arm_key: a.bid_multiplier for a in state.proposed_actions}
    budget_changes = {a.arm_key: a.budget_delta for a in state.proposed_actions}
    platform.update_bids(bid_changes)
    platform.update_budgets(budget_changes)
    # Update local state budgets to mirror platform
    for k, delta in budget_changes.items():
        state.budgets[k] = max(0.0, state.budgets.get(k, 0.0) + delta)
    state.executed_actions.extend(state.proposed_actions)
    state.explainer_log.append({
        "step": state.step,
        "event": "execute_actions",
        "bid_changes": bid_changes,
        "budget_changes": budget_changes,
    })
    state.proposed_actions = []
    return state


def log_node(state: AgentState, out_dir: str = "./agent_runs") -> AgentState:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"run_step_{state.step}.json"), "w") as f:
        json.dump({
            "step": state.step,
            "latest_metrics": {k: asdict(v) for k, v in state.latest_metrics.items()},
            "budgets": state.budgets,
            "explainer": state.explainer_log[-3:],  # last few entries
            "executed": [asdict(a) for a in state.executed_actions[-5:]],
        }, f, indent=2)
    return state

############################
# Wiring (Graph or simple loop)
############################

def build_arms() -> List[Arm]:
    return [
        Arm("search", "kw_high_intent"),
        Arm("search", "kw_generic"),
        Arm("social", "video_a"),
        Arm("social", "video_b"),
        Arm("display", "banner_remarketing"),
    ]

async def simple_loop(num_steps: int = 5,
                      require_human: bool = False,
                      guardrail_blocklist: Optional[List[str]] = None):
    arms = build_arms()
    platform = MockAdPlatform(arms)

    # Initialize state & policy
    state = AgentState()
    for a in arms:
        state.budgets[a.key()] = 200.0
    policy = ThompsonBandit(state)

    cfg = PolicyConfig(require_human_approval=require_human,
                       guardrail_blocklist=guardrail_blocklist or [],
                       daily_budget_cap=2_000.0)
    rng = random.Random(42)

    for _ in range(num_steps):
        fetch_metrics_node(state, platform, policy)
        analyze_node(state, rng, policy)
        propose_actions_node(state, cfg)
        human_approve_node(state, cfg)
        execute_actions_node(state, platform)
        log_node(state)
        await asyncio.sleep(0.1)  # simulate time passing

    print(f"
=== Final budgets (run: {run_label}) ===")
    for k, b in state.budgets.items():
        print(f"{k}: {b:.2f}")

    print("\nRecent actions:")
    for a in state.executed_actions[-10:]:
        print(f"{a.arm_key}: bid×{a.bid_multiplier:.2f}, budgetΔ {a.budget_delta:+.2f} — {a.rationale}")

    print("\nExplainability (last 5 events):")
    for e in state.explainer_log[-5:]:
        print(json.dumps(e, indent=2))

############################
# Optional: LangGraph orchestration
############################

def build_graph(platform: MockAdPlatform, policy: ThompsonBandit, cfg: PolicyConfig):
    if not USE_LANGGRAPH:
        raise RuntimeError("Set USE_LANGGRAPH=True to build the LangGraph graph")
    def node_fetch(s):
        return fetch_metrics_node(s, platform, policy)
    def node_analyze(s):
        return analyze_node(s, random.Random(123), policy)
    def node_propose(s):
        return propose_actions_node(s, cfg)
    def node_approve(s):
        return human_approve_node(s, cfg)
    def node_execute(s):
        return execute_actions_node(s, platform)
    def node_log(s):
        return log_node(s)

    graph = StateGraph(AgentState)
    graph.add_node("fetch", node_fetch)
    graph.add_node("analyze", node_analyze)
    graph.add_node("propose", node_propose)
    graph.add_node("approve", node_approve)
    graph.add_node("execute", node_execute)
    graph.add_node("log", node_log)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "analyze")
    graph.add_edge("analyze", "propose")
    graph.add_edge("propose", "approve")
    graph.add_edge("approve", "execute")
    graph.add_edge("execute", "log")
    graph.add_edge("log", "fetch")  # loop

    return graph.compile()

############################
# Responsible AI checklist (starter)
############################

RESPONSIBLE_AI_README = """
Transparency
- All actions carry a rationale string and are persisted to ./agent_runs/run_step_*.json
Guardrails
- Budget caps enforced; blocklist prevents updates to certain segments.
- Human-in-the-loop optional gate for riskier changes.
Privacy
- This starter does not collect user data. When integrating with platforms, ensure minimal scope OAuth, data retention limits, and secure storage of tokens.
Evaluation
- Track offline metrics (cost, conv) and run A/B or shadow mode before full rollout.
Rollback
- Maintain previous settings and a toggle to revert last N actions.
"""

############################
# Entrypoint
############################

if __name__ == "__main__":
    # Read simple CLI args
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--human", action="store_true")
    p.add_argument("--block", nargs="*", default=[])
    args = p.parse_args()

    print("")
    print("Ad Optimization Agent — Starter (simulation)")
    print("Responsible AI Notes:\n" + RESPONSIBLE_AI_README)

    if USE_LANGGRAPH:
        # Build & run a few iterations of the graph
        arms = build_arms()
        platform = MockAdPlatform(arms)
        state = AgentState()
        for a in arms:
            state.budgets[a.key()] = 200.0
        policy = ThompsonBandit(state)
        cfg = PolicyConfig(require_human_approval=args.human,
                           guardrail_blocklist=args.block,
                           daily_budget_cap=2_000.0)
        app = build_graph(platform, policy, cfg)
        # Run a few cycles
        for _ in range(args.steps):
            state = app.invoke(state)
            time.sleep(0.1)
        print("Done (LangGraph). See ./agent_runs for logs.")
    else:
        asyncio.run(simple_loop(num_steps=args.steps,
                                require_human=args.human,
                                guardrail_blocklist=args.block))


# ===============================
# Connectors: Google Ads / Meta Ads (skeletons)
# ===============================
from typing import Protocol, runtime_checkable

@runtime_checkable
class AdPlatform(Protocol):
    def get_metrics(self) -> Dict[str, Metrics]: ...
    def update_bids(self, changes: Dict[str, float]) -> None: ...
    def update_budgets(self, changes: Dict[str, float]) -> None: ...

class GoogleAdsConnector:
    """Skeleton for Google Ads API integration.
    Replace placeholders with real Google Ads (API vX) client calls.
    Docs: https://developers.google.com/google-ads/api/docs/start
    """
    def __init__(self, credentials_path: str | None = None, customer_id: str | None = None):
        self.customer_id = customer_id
        self.client = None  # load in authenticate()
        self._budgets: Dict[str, float] = {}

    def authenticate(self):
        # from google.ads.googleads.client import GoogleAdsClient
        # self.client = GoogleAdsClient.load_from_storage(credentials_path)
        # validate access
        pass

    def get_metrics(self) -> Dict[str, Metrics]:
        # Query GAQL for last slice window; map rows to Metrics keyed by ad_group_ad or creative id
        # Example keys should match Arm.key() used in the agent.
        # Return a dict like {"search::kw_high_intent": Metrics(...), ...}
        raise NotImplementedError("Implement GAQL query → Metrics mapping")

    def update_bids(self, changes: Dict[str, float]) -> None:
        # Build mutate operations for ad group/ad-level bid modifiers.
        # Clamp at platform limits, handle partial failures.
        # On success, cache current multiplier (optional)
        raise NotImplementedError("Implement bid updates via mutate ops")

    def update_budgets(self, changes: Dict[str, float]) -> None:
        # Build mutate operations for campaign budgets.
        # Consider shared budgets and pacing settings.
        # Update self._budgets cache for transparency.
        raise NotImplementedError("Implement budget updates via mutate ops")

class MetaAdsConnector:
    """Skeleton for Meta Marketing API integration.
    Docs: https://developers.facebook.com/docs/marketing-apis/
    """
    def __init__(self, access_token: str | None = None, account_id: str | None = None):
        self.access_token = access_token
        self.account_id = account_id
        self._budgets: Dict[str, float] = {}

    def authenticate(self):
        # from facebook_business.api import FacebookAdsApi
        # FacebookAdsApi.init(access_token=self.access_token)
        pass

    def get_metrics(self) -> Dict[str, Metrics]:
        # Use AdAccount insights with breakdowns mapping to your Arm keys.
        raise NotImplementedError

    def update_bids(self, changes: Dict[str, float]) -> None:
        # Update ad set bid amounts or bid multipliers.
        raise NotImplementedError

    def update_budgets(self, changes: Dict[str, float]) -> None:
        # Update campaign/ad set daily budgets.
        raise NotImplementedError

# Tip: Swap MockAdPlatform with one of the connectors by duck-typing the same methods.

# ===============================
# Experiment Tracking (MLflow / Weights & Biases)
# ===============================
class ExperimentTracker:
    """Lightweight wrapper that logs to MLflow and/or Weights & Biases if available.
    Usage:
        tracker = ExperimentTracker(project="ad-agent", use_mlflow=True, use_wandb=True)
        with tracker.run(config={"cap": cfg.daily_budget_cap}):
            tracker.log_dict({"step": state.step, ...}, step=state.step)
    """
    def __init__(self, project: str = "ad-agent", use_mlflow: bool = True, use_wandb: bool = True):
        self.project = project
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self._mlflow = None
        self._wandb = None
        # lazy imports
        if self.use_mlflow:
            try:
                import mlflow  # type: ignore
                self._mlflow = mlflow
            except Exception:
                self.use_mlflow = False
        if self.use_wandb:
            try:
                import wandb  # type: ignore
                self._wandb = wandb
            except Exception:
                self.use_wandb = False
        self._active_run = None

    def run(self, config: Dict | None = None):
        class _Ctx:
            def __init__(self, outer: ExperimentTracker, cfg: Dict | None):
                self.outer = outer
                self.cfg = cfg or {}
            def __enter__(self):
                if self.outer.use_mlflow and self.outer._mlflow:
                    self.outer._mlflow.set_experiment(self.outer.project)
                    self.outer._active_run = self.outer._mlflow.start_run()
                    self.outer._mlflow.log_params(self.cfg)
                if self.outer.use_wandb and self.outer._wandb:
                    self.outer._wandb.init(project=self.outer.project, config=self.cfg, reinit=True)
                return self.outer
            def __exit__(self, exc_type, exc, tb):
                if self.outer.use_mlflow and self.outer._mlflow:
                    self.outer._mlflow.end_run()
                if self.outer.use_wandb and self.outer._wandb:
                    self.outer._wandb.finish()
        return _Ctx(self, config)

    def log_dict(self, data: Dict, step: Optional[int] = None):
        if self.use_mlflow and self._mlflow:
            # Flatten shallow dicts
            flat = {}
            for k, v in data.items():
                if isinstance(v, (int, float, str)):
                    flat[k] = v
            self._mlflow.log_metrics({k: v for k, v in flat.items() if isinstance(v, (int, float))}, step=step)
            self._mlflow.log_params({k: v for k, v in flat.items() if isinstance(v, str)})
        if self.use_wandb and self._wandb:
            self._wandb.log(data if step is None else {**data, "step": step})

# --- Minimal integration points (drop-in) ---
# 1) Instantiate near the top of simple_loop():
#    tracker = ExperimentTracker(project="ad-agent", use_mlflow=True, use_wandb=True)
#    with tracker.run(config={"daily_cap": cfg.daily_budget_cap, "require_human": cfg.require_human_approval}):
#        # inside the loop, add tracker.log_dict({...}, step=state.step)
# 2) Add logs after key nodes, e.g. in fetch_metrics_node / propose_actions_node / execute_actions_node.
#    For example, in fetch_metrics_node after updating state:
#        if tracker: tracker.log_dict({"impressions": sum(m.impressions for m in metrics.values()),
#                                     "clicks": sum(m.clicks for m in metrics.values()),
#                                     "conversions": sum(m.conversions for m in metrics.values()),
#                                     "cost": sum(m.cost for m in metrics.values())}, step=state.step)

# ===============================
# Notebook (analysis & charts) — save as notebooks/ad_agent_analysis.py or convert to .ipynb
# ===============================
# %% [markdown]
# # Ad Agent Analysis
# This "notebook-style" script loads JSON logs from ./agent_runs and plots time series for budgets and KPIs.

# %%
if __name__ == "__main__" and False:
    import glob
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt

    run_dir = Path("./agent_runs")
    files = sorted(glob.glob(str(run_dir / "run_step_*.json")))
    steps = []
    total_cost = []
    total_clicks = []
    total_convs = []
    budgets_over_time: Dict[str, List[float]] = {}

    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)
        step = d.get("step")
        steps.append(step)
        lm = d.get("latest_metrics", {})
        cost = sum(v.get("cost", 0.0) for v in lm.values())
        clicks = sum(v.get("clicks", 0) for v in lm.values())
        convs = sum(v.get("conversions", 0) for v in lm.values())
        total_cost.append(cost)
        total_clicks.append(clicks)
        total_convs.append(convs)
        budgets = d.get("budgets", {})
        for k, b in budgets.items():
            budgets_over_time.setdefault(k, []).append(b)

    # Plot totals
    plt.figure()
    plt.plot(steps, total_cost)
    plt.title("Total Cost per Step")
    plt.xlabel("Step")
    plt.ylabel("Cost")
    plt.show()

    plt.figure()
    plt.plot(steps, total_clicks)
    plt.title("Total Clicks per Step")
    plt.xlabel("Step")
    plt.ylabel("Clicks")
    plt.show()

    plt.figure()
    plt.plot(steps, total_convs)
    plt.title("Total Conversions per Step")
    plt.xlabel("Step")
    plt.ylabel("Conversions")
    plt.show()

    # Plot budgets for a few arms
    for k, series in list(budgets_over_time.items())[:5]:
        plt.figure()
        plt.plot(steps, series)
        plt.title(f"Budget over time — {k}")
        plt.xlabel("Step")
        plt.ylabel("Budget")
        plt.show()

# ===============================
# Tests (pytest) — save as tests/test_agent.py
# ===============================
TESTS = r"""
import json
from ad_agent_starter import (
    AgentState, ThompsonBandit, MockAdPlatform, build_arms,
    fetch_metrics_node, analyze_node, propose_actions_node,
    human_approve_node, execute_actions_node, PolicyConfig
)


def test_bandit_updates_and_actions(tmp_path):
    arms = build_arms()
    platform = MockAdPlatform(arms, seed=123)
    state = AgentState()
    for a in arms:
        state.budgets[a.key()] = 200.0
    policy = ThompsonBandit(state)
    cfg = PolicyConfig(daily_budget_cap=1_000.0)

    # One iteration
    fetch_metrics_node(state, platform, policy)
    analyze_node(state, __import__("random").Random(0), policy)
    propose_actions_node(state, cfg)
    # Sanity: we must have proposed actions for every arm
    assert len(state.proposed_actions) == len(state.budgets)

    # Execute
    execute_actions_node(state, platform)
    # Budgets should have changed but remain non-negative
    assert all(b >= 0 for b in state.budgets.values())


def test_budget_cap_enforced():
    arms = build_arms()
    platform = MockAdPlatform(arms, seed=1)
    state = AgentState()
    for a in arms:
        state.budgets[a.key()] = 2_000.0
    policy = ThompsonBandit(state)
    cfg = PolicyConfig(daily_budget_cap=3_000.0)

    fetch_metrics_node(state, platform, policy)
    analyze_node(state, __import__("random").Random(0), policy)
    propose_actions_node(state, cfg)

    projected_total = sum(state.budgets.values()) + sum(a.budget_delta for a in state.proposed_actions)
    assert projected_total <= cfg.daily_budget_cap + 1e-6
"""

# To materialize tests, write TESTS to tests/test_agent.py on first run or provide separate files in your project.
