"""Task/scenario definitions: the multi-step agentic tasks the harness runs,
grouped by chain length (1, 3, 5 — configs/experiment.yaml).

A task is a scripted scenario, not a free-form prompt: it specifies the user
goal, the sequence of tool calls a correct agent would make
(expected_tool_sequence — checked step-by-step by executor.run_task, not
just left implicit), and, for tasks with an injected_error_at_step, which
step should have its first attempt forced to fail so we can score recovery
behavior specifically.

Two task sources, kept deliberately distinct — NOT because one is a
placeholder for the other, but because they serve different pipeline stages
and mixing them would confound the research question:

  - build_synthetic_tasks() — 30-50 hand-designed scenarios per chain length,
    built on harness.tools.build_demo_registry()'s 6-tool vocabulary
    (get_weather, calculator, search_knowledge_base, get_current_time,
    convert_temperature, compare_numbers). This is the PRIMARY eval set for
    all three chain lengths (1, 3, 5) — see "why synthetic for eval" below.

  - load_tasks(chain_length, source=...) — the loader evaluation/run_eval.py
    (and this module's own tests) actually call:
      - source="synthetic" (default): build_synthetic_tasks() above.
      - source="glaive_train" / "glaive_test": real glaive-function-calling-v2
        examples from data/splits/{train,test}.jsonl (adbench.data.prepare),
        run through harness.tools.build_glaive_registry(). Single-step only
        (chain_length must be 1) — data/prepare.py found essentially no
        natural multi-step chains in the raw dataset (8 of 112,960 examples;
        see its module docstring). This is a genuinely thin wrapper: each
        JSONL row already carries every TaskSpec field.

WHY EVAL USES THE SYNTHETIC SET, NOT GLAIVE, FOR ALL THREE CHAIN LENGTHS:
The research question is whether success rate degrades *as chain length
increases* (1 vs 3 vs 5). Chain length has to be the only thing that varies
across that comparison. If chain_length=1 eval used real glaive data (its
own 8-tool vocabulary, its own phrasing style) while chain_length=3/5 used
the synthetic 6-tool set, any measured "degradation" would be confounded
with a simultaneous tool-domain shift — the model would be failing (or
succeeding) partly because it's facing unfamiliar tools, not necessarily
because the chain got longer. Holding the tool vocabulary fixed across all
three chain lengths isolates chain length as the manipulated variable.

This does mean the primary eval set is NOT the distribution the models are
fine-tuned on (that's glaive, via data/prepare.py, used by training/ in Part
2) — eval is deliberately measuring generalization to a fixed-but-unfamiliar
tool vocabulary, consistently across conditions. `source="glaive_test"`
exists as a secondary, single-step-only, in-distribution data point for
anyone who wants it (e.g. sanity-checking that a fine-tuned model didn't
also regress on the exact data it trained on) — it is not part of the
chain-length comparison. See README.md's "Why synthetic tasks for
multi-step eval" section for the full writeup.

KNOWN LIMITATION: the harness (executor.run_task) currently grades whether
the model called the *correct tool, in the correct order* — it does not
verify that a step's *arguments* actually derive from an earlier step's
result. So a "genuinely sequential, later-depends-on-earlier" scenario (as
designed below) is graded the same way as a scenario where the model
happened to guess the right tool names without ever reading the
intermediate results. This is a real scope boundary, not an oversight:
verifying argument-level dependency would need either numeric-value
extraction from arbitrary model phrasing or an LLM-judge, both of which are
significant additions belonging to evaluation/, not here. See
tests/test_tasks.py's dependency-feasibility tests, which use a scripted
model that actually reads and computes from prior results, for evidence
that the designed dependencies are real and completable — just not yet
independently *enforced* by the grader.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from adbench.harness.tools import (
    KNOWLEDGE_BASE,
    TIME_TABLE,
    WEATHER_TABLE,
    ToolRegistry,
    build_demo_registry,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    chain_length: int
    user_goal: str
    expected_tool_sequence: list[str] = field(default_factory=list)
    injected_error_at_step: int | None = None


# --------------------------------------------------------------------------
# Synthetic task generation. Built from small templates x parameter lists
# rather than one giant hand-typed list: keeps 120+ scenarios maintainable,
# and pulls city/timezone/topic vocabulary directly from tools.py's tables
# (WEATHER_TABLE/TIME_TABLE/KNOWLEDGE_BASE) so generated tasks can never
# drift out of sync with what the tools actually support.
# --------------------------------------------------------------------------

_CITIES = list(WEATHER_TABLE)          # 10 cities
_ZONES = list(TIME_TABLE)              # 8 timezones
_TOPICS = list(KNOWLEDGE_BASE)         # 7 topics

_CALC_EXPRESSIONS = [
    "15 * 4", "100 / 5", "9 + 16", "50 - 12", "7 * 8 - 3",
    "144 / 12 + 1", "6 * 6 + 6", "200 - 45",
]

_TEMP_CONVERSIONS = [  # (value, from_unit, to_unit)
    (98.6, "F", "C"), (0, "C", "F"), (300, "K", "C"),
    (32, "F", "C"), (100, "C", "K"), (273.15, "K", "C"),
]

_NUMBER_PAIRS = [  # (a, b)
    (42, 17), (100, 250), (7.5, 7.5), (63, 12), (1000, 999), (5, 55), (81, 8),
]


def _slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_")


def _expand(
    id_prefix: str,
    tool_sequence: list[str],
    goal_fn: Any,
    param_list: list[dict[str, Any]],
) -> list[TaskSpec]:
    """Instantiate one template across its parameter variations."""
    return [
        TaskSpec(
            task_id=f"{id_prefix}-{i}",
            chain_length=len(tool_sequence),
            user_goal=goal_fn(p),
            expected_tool_sequence=list(tool_sequence),
        )
        for i, p in enumerate(param_list)
    ]


def _gen_chain1_tasks() -> list[TaskSpec]:
    """30-50 single-step tasks: chain_length=1 has no "dependency" concept
    (nothing precedes the one step) — variety here just means covering every
    tool with several different argument values."""
    tasks: list[TaskSpec] = []

    for city in _CITIES:
        tasks.append(TaskSpec(
            task_id=f"c1-weather-{_slug(city)}",
            chain_length=1,
            user_goal=f"What's the weather in {city.title()}?",
            expected_tool_sequence=["get_weather"],
        ))
    for i, expr in enumerate(_CALC_EXPRESSIONS):
        tasks.append(TaskSpec(
            task_id=f"c1-calc-{i}",
            chain_length=1,
            user_goal=f"What is {expr}?",
            expected_tool_sequence=["calculator"],
        ))
    for topic in _TOPICS:
        tasks.append(TaskSpec(
            task_id=f"c1-kb-{_slug(topic)}",
            chain_length=1,
            user_goal=f"Can you look up '{topic}' in the knowledge base?",
            expected_tool_sequence=["search_knowledge_base"],
        ))
    for zone in _ZONES:
        tasks.append(TaskSpec(
            task_id=f"c1-time-{zone}",
            chain_length=1,
            user_goal=f"What time is it in {zone.upper()}?",
            expected_tool_sequence=["get_current_time"],
        ))
    for i, (value, from_u, to_u) in enumerate(_TEMP_CONVERSIONS):
        tasks.append(TaskSpec(
            task_id=f"c1-convert-{i}",
            chain_length=1,
            user_goal=f"Convert {value} degrees {from_u} to {to_u}.",
            expected_tool_sequence=["convert_temperature"],
        ))
    for i, (a, b) in enumerate(_NUMBER_PAIRS):
        tasks.append(TaskSpec(
            task_id=f"c1-compare-{i}",
            chain_length=1,
            user_goal=f"Which is bigger, {a} or {b}?",
            expected_tool_sequence=["compare_numbers"],
        ))
    return tasks


def _gen_chain3_tasks() -> list[TaskSpec]:
    """8 templates x 5 parameter variations = 40 three-step tasks. Each
    template is written so a later step genuinely follows from an earlier
    one (comparing two things just measured, converting a reading just
    fetched, computing from a value just converted) — not independent asks
    bundled into one message. See the module docstring's "KNOWN LIMITATION"
    on how far that dependency is currently verified vs just designed-in."""
    tasks: list[TaskSpec] = []

    tasks += _expand(
        "c3-weather_convert_compare",
        ["get_weather", "convert_temperature", "compare_numbers"],
        lambda p: (
            f"Check the weather in {p['city'].title()}, convert that temperature to "
            f"Celsius, then tell me whether it's higher or lower than {p['threshold']}°C."
        ),
        [
            {"city": _CITIES[0], "threshold": 15}, {"city": _CITIES[1], "threshold": 20},
            {"city": _CITIES[2], "threshold": 25}, {"city": _CITIES[3], "threshold": 10},
            {"city": _CITIES[4], "threshold": 30},
        ],
    )
    tasks += _expand(
        "c3-weather_weather_compare",
        ["get_weather", "get_weather", "compare_numbers"],
        lambda p: (
            f"Check the weather in {p['a'].title()} and in {p['b'].title()}, "
            "then tell me which one is warmer."
        ),
        [
            {"a": "paris", "b": "tokyo"}, {"a": "cairo", "b": "london"},
            {"a": "berlin", "b": "sydney"}, {"a": "mumbai", "b": "toronto"},
            {"a": "dubai", "b": "san francisco"},
        ],
    )
    tasks += _expand(
        "c3-calc_calc_compare",
        ["calculator", "calculator", "compare_numbers"],
        lambda p: (
            f"Compute {p['a']}, separately compute {p['b']}, "
            "then tell me which result is larger."
        ),
        [
            {"a": _CALC_EXPRESSIONS[0], "b": _CALC_EXPRESSIONS[2]},
            {"a": _CALC_EXPRESSIONS[1], "b": _CALC_EXPRESSIONS[3]},
            {"a": _CALC_EXPRESSIONS[4], "b": _CALC_EXPRESSIONS[5]},
            {"a": _CALC_EXPRESSIONS[6], "b": _CALC_EXPRESSIONS[7]},
            {"a": _CALC_EXPRESSIONS[0], "b": _CALC_EXPRESSIONS[6]},
        ],
    )
    tasks += _expand(
        "c3-weather_calc_compare",
        ["get_weather", "calculator", "compare_numbers"],
        lambda p: (
            f"Check the weather in {p['city'].title()}, separately compute {p['expr']}, "
            "then compare the temperature reading to that calculated value."
        ),
        [
            {"city": _CITIES[5], "expr": _CALC_EXPRESSIONS[0]},
            {"city": _CITIES[6], "expr": _CALC_EXPRESSIONS[1]},
            {"city": _CITIES[7], "expr": _CALC_EXPRESSIONS[2]},
            {"city": _CITIES[8], "expr": _CALC_EXPRESSIONS[3]},
            {"city": _CITIES[9], "expr": _CALC_EXPRESSIONS[4]},
        ],
    )
    tasks += _expand(
        "c3-convert_convert_compare",
        ["convert_temperature", "convert_temperature", "compare_numbers"],
        lambda p: (
            f"Convert {p['a'][0]} degrees {p['a'][1]} to Celsius, convert {p['b'][0]} "
            f"degrees {p['b'][1]} to Celsius as well, then tell me which is warmer."
        ),
        [
            {"a": _TEMP_CONVERSIONS[0], "b": _TEMP_CONVERSIONS[1]},
            {"a": _TEMP_CONVERSIONS[2], "b": _TEMP_CONVERSIONS[3]},
            {"a": _TEMP_CONVERSIONS[4], "b": _TEMP_CONVERSIONS[5]},
            {"a": _TEMP_CONVERSIONS[0], "b": _TEMP_CONVERSIONS[4]},
            {"a": _TEMP_CONVERSIONS[1], "b": _TEMP_CONVERSIONS[5]},
        ],
    )
    tasks += _expand(
        "c3-weather_convert_calc",
        ["get_weather", "convert_temperature", "calculator"],
        lambda p: (
            f"Check the weather in {p['city'].title()}, convert that reading to Celsius, "
            "then compute how many degrees above freezing (0°C) that converted "
            "temperature is."
        ),
        [{"city": c} for c in _CITIES[:5]],
    )
    tasks += _expand(
        "c3-kb_weather_convert",
        ["search_knowledge_base", "get_weather", "convert_temperature"],
        lambda p: (
            f"Look up '{p['topic']}' in the knowledge base, then check the weather in "
            f"{p['city'].title()} and convert that temperature to Celsius."
        ),
        [{"topic": t, "city": c} for t, c in zip(_TOPICS[:5], _CITIES[:5], strict=False)],
    )
    tasks += _expand(
        "c3-time_kb_calc",
        ["get_current_time", "search_knowledge_base", "calculator"],
        lambda p: (
            f"Tell me the current time in {p['zone'].upper()}, look up '{p['topic']}' "
            f"in the knowledge base, then compute {p['expr']}."
        ),
        [
            {"zone": z, "topic": t, "expr": e}
            for z, t, e in zip(_ZONES[:5], _TOPICS[:5], _CALC_EXPRESSIONS[:5], strict=False)
        ],
    )
    return tasks


def _gen_chain5_tasks() -> list[TaskSpec]:
    """7 templates x 5 parameter variations = 35 five-step tasks."""
    tasks: list[TaskSpec] = []

    tasks += _expand(
        "c5-weather_convert_weather_convert_compare",
        ["get_weather", "convert_temperature", "get_weather", "convert_temperature", "compare_numbers"],
        lambda p: (
            f"Check the weather in {p['a'].title()} and convert it to Celsius. Then check "
            f"the weather in {p['b'].title()} and convert that to Celsius too. Finally, "
            "tell me which city is warmer."
        ),
        [
            {"a": "paris", "b": "tokyo"}, {"a": "cairo", "b": "berlin"},
            {"a": "london", "b": "sydney"}, {"a": "mumbai", "b": "dubai"},
            {"a": "toronto", "b": "san francisco"},
        ],
    )
    tasks += _expand(
        "c5-calc_calc_compare_convert_calc",
        ["calculator", "calculator", "compare_numbers", "convert_temperature", "calculator"],
        lambda p: (
            f"Compute {p['a']}. Compute {p['b']}. Compare the two results. Convert "
            f"{p['temp'][0]} degrees {p['temp'][1]} to Celsius. Finally compute {p['c']}."
        ),
        [
            {"a": _CALC_EXPRESSIONS[0], "b": _CALC_EXPRESSIONS[1], "temp": _TEMP_CONVERSIONS[0], "c": _CALC_EXPRESSIONS[2]},
            {"a": _CALC_EXPRESSIONS[3], "b": _CALC_EXPRESSIONS[4], "temp": _TEMP_CONVERSIONS[1], "c": _CALC_EXPRESSIONS[5]},
            {"a": _CALC_EXPRESSIONS[6], "b": _CALC_EXPRESSIONS[7], "temp": _TEMP_CONVERSIONS[2], "c": _CALC_EXPRESSIONS[0]},
            {"a": _CALC_EXPRESSIONS[1], "b": _CALC_EXPRESSIONS[2], "temp": _TEMP_CONVERSIONS[3], "c": _CALC_EXPRESSIONS[4]},
            {"a": _CALC_EXPRESSIONS[5], "b": _CALC_EXPRESSIONS[6], "temp": _TEMP_CONVERSIONS[4], "c": _CALC_EXPRESSIONS[7]},
        ],
    )
    tasks += _expand(
        "c5-kb_weather_convert_weather_convert",
        ["search_knowledge_base", "get_weather", "convert_temperature", "get_weather", "convert_temperature"],
        lambda p: (
            f"Look up '{p['topic']}', then check the weather in {p['a'].title()} and "
            f"convert it to Celsius, then check the weather in {p['b'].title()} and "
            "convert that to Celsius too."
        ),
        [
            {"topic": _TOPICS[0], "a": "paris", "b": "cairo"},
            {"topic": _TOPICS[1], "a": "tokyo", "b": "london"},
            {"topic": _TOPICS[2], "a": "berlin", "b": "sydney"},
            {"topic": _TOPICS[3], "a": "mumbai", "b": "toronto"},
            {"topic": _TOPICS[4], "a": "dubai", "b": "san francisco"},
        ],
    )
    tasks += _expand(
        "c5-time_weather_convert_calc_compare",
        ["get_current_time", "get_weather", "convert_temperature", "calculator", "compare_numbers"],
        lambda p: (
            f"Tell me the time in {p['zone'].upper()}. Check the weather in "
            f"{p['city'].title()} and convert it to Celsius. Compute {p['expr']}. Then "
            "compare the converted temperature to that calculation result."
        ),
        [
            {"zone": z, "city": c, "expr": e}
            for z, c, e in zip(_ZONES[:5], _CITIES[:5], _CALC_EXPRESSIONS[:5], strict=False)
        ],
    )
    tasks += _expand(
        "c5-weather_convert_compare_kb_calc",
        ["get_weather", "convert_temperature", "compare_numbers", "search_knowledge_base", "calculator"],
        lambda p: (
            f"Check the weather in {p['city'].title()}, convert it to Celsius, and "
            f"compare it to {p['threshold']}°C. Then look up '{p['topic']}' for "
            f"some context, and compute {p['expr']}."
        ),
        [
            {"city": _CITIES[5], "threshold": 18, "topic": _TOPICS[5], "expr": _CALC_EXPRESSIONS[0]},
            {"city": _CITIES[6], "threshold": 22, "topic": _TOPICS[6], "expr": _CALC_EXPRESSIONS[1]},
            {"city": _CITIES[7], "threshold": 12, "topic": _TOPICS[0], "expr": _CALC_EXPRESSIONS[2]},
            {"city": _CITIES[8], "threshold": 28, "topic": _TOPICS[1], "expr": _CALC_EXPRESSIONS[3]},
            {"city": _CITIES[9], "threshold": 16, "topic": _TOPICS[2], "expr": _CALC_EXPRESSIONS[4]},
        ],
    )
    tasks += _expand(
        "c5-calc_convert_calc_compare_calc",
        ["calculator", "convert_temperature", "calculator", "compare_numbers", "calculator"],
        lambda p: (
            f"Compute {p['a']}. Convert {p['temp'][0]} degrees {p['temp'][1]} to Celsius. "
            f"Compute {p['b']}. Compare the two calculated results. Finally compute {p['c']}."
        ),
        [
            {"a": _CALC_EXPRESSIONS[0], "temp": _TEMP_CONVERSIONS[5], "b": _CALC_EXPRESSIONS[1], "c": _CALC_EXPRESSIONS[2]},
            {"a": _CALC_EXPRESSIONS[3], "temp": _TEMP_CONVERSIONS[0], "b": _CALC_EXPRESSIONS[4], "c": _CALC_EXPRESSIONS[5]},
            {"a": _CALC_EXPRESSIONS[6], "temp": _TEMP_CONVERSIONS[1], "b": _CALC_EXPRESSIONS[7], "c": _CALC_EXPRESSIONS[0]},
            {"a": _CALC_EXPRESSIONS[2], "temp": _TEMP_CONVERSIONS[2], "b": _CALC_EXPRESSIONS[3], "c": _CALC_EXPRESSIONS[6]},
            {"a": _CALC_EXPRESSIONS[5], "temp": _TEMP_CONVERSIONS[3], "b": _CALC_EXPRESSIONS[7], "c": _CALC_EXPRESSIONS[1]},
        ],
    )
    tasks += _expand(
        "c5-weather_weather_compare_convert_compare",
        ["get_weather", "get_weather", "compare_numbers", "convert_temperature", "compare_numbers"],
        lambda p: (
            f"Check the weather in {p['a'].title()} and {p['b'].title()}, compare them. "
            f"Convert {p['a'].title()}'s reading to Celsius. Then compare that Celsius "
            f"value to {p['threshold']}°C."
        ),
        [
            {"a": "paris", "b": "tokyo", "threshold": 14},
            {"a": "cairo", "b": "london", "threshold": 26},
            {"a": "berlin", "b": "mumbai", "threshold": 19},
            {"a": "sydney", "b": "toronto", "threshold": 21},
            {"a": "dubai", "b": "san francisco", "threshold": 24},
        ],
    )
    return tasks


def _assign_injected_errors(tasks: list[TaskSpec]) -> list[TaskSpec]:
    """Deterministically mark every 4th task's first attempt at one step to
    fail (configs/experiment.yaml: harness.inject_errors), cycling the
    injected step position so error-recovery gets exercised at different
    points in the chain, not always step 0."""
    out = []
    for i, task in enumerate(tasks):
        if i % 4 == 0:
            step = (i // 4) % task.chain_length
            out.append(replace(task, injected_error_at_step=step))
        else:
            out.append(task)
    return out


def _make_synthetic_scenarios() -> dict[int, list[TaskSpec]]:
    return {
        1: _assign_injected_errors(_gen_chain1_tasks()),
        3: _assign_injected_errors(_gen_chain3_tasks()),
        5: _assign_injected_errors(_gen_chain5_tasks()),
    }


def build_synthetic_tasks(
    chain_length: int, registry: ToolRegistry | None = None
) -> list[TaskSpec]:
    """The primary eval task set (see module docstring for why eval uses
    this instead of real glaive data). Validates every expected tool name
    actually exists in `registry` (defaults to the demo registry these
    scenarios were written against), so a typo here fails fast instead of
    surfacing as a confusing UnknownToolError deep in a test run.
    """
    registry = registry or build_demo_registry()
    scenarios = _make_synthetic_scenarios()
    if chain_length not in scenarios:
        raise ValueError(
            f"No synthetic scenarios defined for chain_length={chain_length}; "
            f"supported: {sorted(scenarios)}"
        )

    tasks = scenarios[chain_length]
    for task in tasks:
        for name in task.expected_tool_sequence:
            if not registry.has(name):
                raise ValueError(
                    f"Task {task.task_id!r} expects unknown tool {name!r}."
                )
    return tasks


def load_tasks(
    chain_length: int,
    source: str = "synthetic",
    config_path: str | Path | None = None,
) -> list[TaskSpec]:
    """The task loader evaluation/run_eval.py calls.

    source="synthetic" (default): build_synthetic_tasks(chain_length) — the
        6-tool demo vocabulary, held constant across chain_length in
        {1, 3, 5}. This is the primary eval axis for the "does success rate
        degrade with chain length" comparison — see module docstring for why
        eval doesn't use real glaive data for this.

    source="glaive_train" / "glaive_test": real glaive-function-calling-v2
        examples, read directly from data/splits/{train,test}.jsonl
        (adbench.data.prepare) and converted 1:1 into TaskSpecs (each JSONL
        row already carries task_id/chain_length/user_goal/
        expected_tool_sequence/injected_error_at_step). Tools referenced are
        from harness.tools.build_glaive_registry(), not the demo registry —
        callers must pass that registry to executor.run_task(), not the
        default demo one. Only chain_length=1 is available (data/prepare.py
        found essentially no natural multi-step chains in the raw dataset).
        This is NOT how Part 2's SFT/KD trainer consumes the glaive data (it
        reads the JSONL directly for teacher-forced loss on real text, not
        an interactive harness rollout) — this path exists for harness-side
        use: a real, in-distribution, single-step data point kept separate
        from the primary synthetic eval axis, and for sanity-checking that
        build_glaive_registry()'s tools work end-to-end through the same
        executor.run_task() loop the synthetic tasks use.

    `config_path` overrides where configs/data.yaml is read from (mainly for
    tests, so the glaive-backed path is testable against fixture data
    without depending on the real data/splits/*.jsonl existing) — defaults
    to this repo's real configs/data.yaml.

    Raises FileNotFoundError for the glaive sources if
    `python -m adbench.data.prepare` hasn't been run yet, and ValueError for
    an unrecognized `source` or an unsupported chain_length/source pairing.
    """
    if source == "synthetic":
        return build_synthetic_tasks(chain_length)

    if source in ("glaive_train", "glaive_test"):
        if chain_length != 1:
            raise ValueError(
                f"source={source!r} only has chain_length=1 data (see "
                f"data/prepare.py's module docstring); got chain_length={chain_length}."
            )
        # Deferred imports: keeps a plain `import adbench.harness.tasks` from
        # requiring pyyaml unless this glaive-backed path is actually used.
        from adbench.data.prepare import load_config, read_jsonl

        resolved_config_path = Path(config_path) if config_path else (_REPO_ROOT / "configs" / "data.yaml")
        config = load_config(resolved_config_path)
        path_key = "train_path" if source == "glaive_train" else "test_path"
        # Output paths in data.yaml are relative to the repo root, same
        # convention as adbench.data.prepare._resolve().
        path = _REPO_ROOT / config["output"][path_key]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist yet — run `python -m adbench.data.prepare` first."
            )
        return [
            TaskSpec(
                task_id=r["task_id"],
                chain_length=r["chain_length"],
                user_goal=r["user_goal"],
                expected_tool_sequence=r["expected_tool_sequence"],
                injected_error_at_step=r["injected_error_at_step"],
            )
            for r in read_jsonl(path)
        ]

    raise ValueError(
        f"Unknown source {source!r}; expected 'synthetic', 'glaive_train', or 'glaive_test'."
    )
