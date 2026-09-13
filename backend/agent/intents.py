"""Coding-request understanding: intent, parameters, complexity. No model.

Stage 1: anchored rules (microseconds). Stage 2: TF-IDF + Logistic Regression trained
on the seed phrasings below, for wordings the rules miss; its cross-validated accuracy
is stored next to the model. Parameters (paths, symbols, line ranges, globs, rename
pairs, test targets) are extracted with regexes, never guessed by an LLM.

Deterministic intents are answered by one tool call. Agent intents run the controlled
loop, which decides per step whether a tool, NLP, ML or an LLM is needed.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field

from backend.config import DATA_DIR

MODEL_PATH = DATA_DIR / "models" / "coding_intent_clf.joblib"
META_PATH = DATA_DIR / "models" / "coding_intent_clf.json"

DETERMINISTIC = {
    "file_exists", "list_directory", "read_file", "file_info", "search_files", "search_code",
    "search_symbols", "git_status", "git_diff", "git_log", "run_tests", "lint", "syntax_check",
    "typecheck", "parse_data_file", "system_status", "memory_query", "list_checkpoints",
    "rollback", "project_overview", "rename_file", "delete_file", "calculate", "file_outline",
}
AGENT = {"explain_code", "fix_bug", "analyze_error", "implement", "refactor", "rename_symbol",
         "write_tests", "review", "create_file", "general_coding"}
INTENTS = sorted(DETERMINISTIC | AGENT)

_P = r"[\w./\\-]+"  # a path-ish token
RULES: list[tuple[str, re.Pattern, float]] = [
    # Order matters: specific before general.
    ("rollback", re.compile(r"^\s*(?:undo|revert|roll ?back)\b", re.I), 0.95),
    ("list_checkpoints", re.compile(r"\b(?:list|show)\b.{0,15}\bcheckpoints?\b", re.I), 0.95),
    ("memory_query", re.compile(r"\bwhat (?:was|were) (?:i|we) (?:working on|doing)\b|\bunfinished (?:tasks?|work)\b|"
                                r"\bwhat did (?:we|i) decide\b|\b(?:show|list) (?:my |the )?(?:memory|decisions|past tasks)\b", re.I), 0.93),
    ("system_status", re.compile(r"\b(?:cpu|ram|memory) (?:usage|load|use)\b|\bwhat'?s using (?:the )?(?:cpu|ram|memory)\b|"
                                 r"\bsystem (?:resources|status|load)\b|\btop processes\b", re.I), 0.94),
    ("git_status", re.compile(r"\bgit status\b|\bwhat (?:files )?(?:have )?changed\b|\b(?:which|what) files are (?:modified|changed|staged)\b|"
                              r"\buncommitted changes\b", re.I), 0.95),
    ("git_diff", re.compile(r"\bgit diff\b|\bshow (?:me )?(?:the )?diff\b", re.I), 0.95),
    ("git_log", re.compile(r"\bgit log\b|\brecent commits\b|\b(?:last|latest) \d* ?commits?\b|\bcommit history\b", re.I), 0.95),
    ("review", re.compile(r"\breview\b.{0,30}\b(?:changes|diff|code|pr|pull request|commit|file|\.py)\b|\bcode review\b", re.I), 0.92),
    ("write_tests", re.compile(r"\b(?:write|add|create|generate)\b.{0,20}\b(?:unit |integration )?tests?\b(?! (?:pass|fail))|"
                               r"\btest coverage for\b", re.I), 0.9),
    ("run_tests", re.compile(r"\b(?:run|execute|rerun|re-run)\b.{0,20}\b(?:tests?|pytest|test suite|specs?)\b|^\s*pytest\b|"
                             r"\bdo (?:the )?tests pass\b", re.I), 0.95),
    ("typecheck", re.compile(r"\btype ?check\b|\brun tsc\b|\btsc --noemit\b", re.I), 0.94),
    ("lint", re.compile(r"\b(?:lint|pyflakes|flake8|unused imports?|undefined names?)\b", re.I), 0.93),
    ("syntax_check", re.compile(r"\bsyntax (?:check|error|errors)\b|\bcheck (?:the )?syntax\b|\bdoes .{1,60} compile\b|\bis .{1,60} valid (?:python|json)\b", re.I), 0.93),
    ("rename_symbol", re.compile(r"\brename (?:the )?(?:function|class|method|variable|symbol)\s+`?[A-Za-z_]\w*`?\s+to\s+`?[A-Za-z_]\w*`?", re.I), 0.94),
    ("rename_file", re.compile(rf"\b(?:rename|move) (?:the )?(?:file )?{_P}\.\w+\s+to\s+{_P}", re.I), 0.93),
    ("delete_file", re.compile(rf"\b(?:delete|remove) (?:the )?file\s+{_P}", re.I), 0.93),
    ("fix_bug", re.compile(r"\b(?:fix|repair|debug|solve|resolve)\b.{0,50}\b(?:bug|tests?|failing|failure|errors?|crash|traceback|exception|issue|broken)\b|"
                           r"\bmake (?:the )?tests? pass\b|\btests? (?:are )?failing\b", re.I), 0.92),
    ("analyze_error", re.compile(r"\btraceback \(most recent call last\)|\bwhy (?:does|is|do|am)\b.{0,60}\b(?:error|fail|failing|crash|exception|raise)|"
                                 r"\bwhat (?:does|causes) (?:this|the) (?:error|exception)\b", re.I), 0.9),
    ("refactor", re.compile(r"\brefactor\b|\bextract (?:a |the )?(?:function|method|class)\b|\bsimplify (?:the )?(?:function|code|class|method)\b", re.I), 0.9),
    ("create_file", re.compile(rf"\bcreate (?:a |an |the )?(?:new )?file\s+{_P}", re.I), 0.9),
    ("implement", re.compile(r"^\s*(?:please\s+)?implement\b|\b(?:implement|add|build|create)\b.{0,40}\b(?:function|method|class|endpoint|feature|route|component|option|flag|parameter|validation|support for|cache|caching)\b", re.I), 0.88),
    ("file_outline", re.compile(r"\boutline (?:of )?" + _P + r"\.\w+|\b(?:list (?:the )?(?:functions|classes|symbols)|what (?:functions|classes) are) (?:in|of) " + _P, re.I), 0.93),
    ("search_symbols", re.compile(r"\bwhere (?:is|are) (?:the )?(?:function|class|method|def|symbol)?\s*`?[A-Za-z_][\w.]*`?\s+(?:defined|declared|implemented)\b|"
                                  r"\b(?:find|go to|show) (?:the )?definition of\b", re.I), 0.94),
    ("search_code", re.compile(r"\bgrep\b|\bsearch (?:the )?(?:code|codebase|repo|project)\b|\bfind (?:all )?(?:usages|uses|references|occurrences) of\b|"
                               r"\bwhere (?:is|are) .{1,40} (?:used|called|referenced)\b|\bwhich files (?:contain|mention|use)\b", re.I), 0.93),
    ("search_files", re.compile(r"\bfind (?:all )?(?:the )?files?\b|\b(?:list|show) (?:all )?(?:\*?\.\w+|python|typescript|json|markdown|test) files\b|"
                                r"\bfiles? (?:named|called|matching)\b", re.I), 0.92),
    # Built by concatenation, not rf-strings: an f-string turns a quantifier like {0,30} into "(0, 30)".
    ("file_exists", re.compile(r"\b(?:does|is there (?:a|any)) (?:file )?" + _P + r"\s*(?:file )?exists?\b|\bdoes .{1,60} exist\b", re.I), 0.94),
    ("file_info", re.compile(r"\bhow (?:big|large|many lines)\b.{0,30}" + _P + r"\.\w+|\b(?:size|line count) of " + _P, re.I), 0.93),
    ("parse_data_file", re.compile(r"\b(?:parse|validate|load|read)\b.{0,20}[\w./\\-]+\.(?:json|csv)\b", re.I), 0.93),
    ("list_directory", re.compile(r"\b(?:list|show|what(?:'s| is) in)\b.{0,12}\b(?:the )?(?:directory|folder|dir)\b|^\s*ls\b|"
                                  r"\blist (?:the )?(?:contents|files) (?:of|in)\b", re.I), 0.93),
    ("read_file", re.compile(r"\b(?:show|open|read|cat|print|display)\b(?: me)?(?: the)?(?: contents of)?(?: file)?\s+`?" + _P + r"\.\w+`?|"
                             r"\b(?:show|print|display|read)\b.{0,10}\blines?\s+\d+.{0,25}" + _P + r"\.\w+", re.I), 0.93),
    ("project_overview", re.compile(r"\b(?:summari[sz]e|overview of|describe|explain) (?:the |this )?(?:project|codebase|repo|repository|architecture)\b|"
                                    r"\bwhat is this (?:project|codebase|repo)\b|\bproject structure\b", re.I), 0.9),
    ("explain_code", re.compile(r"\b(?:explain|what does|how does|walk me through|describe)\b.{0,60}\b(?:function|class|method|module|file|code|work|\.py|\.tsx?|\(\))", re.I), 0.88),
]


def rule_intent(text: str) -> tuple[str | None, float, str]:
    from backend.tools.deterministic import looks_like_math

    t = (text or "").strip()
    if looks_like_math(t):
        return "calculate", 0.97, "math_gate"
    for intent, rx, conf in RULES:
        if rx.search(t):
            return intent, conf, f"rule:{intent}"
    return None, 0.0, "no_rule"


SEED: dict[str, list[str]] = {
    "file_exists": ["does config.json exist", "is there a README in this repo", "check if setup.py exists"],
    "list_directory": ["list the backend folder", "what is inside src", "show me the directory tree of tests"],
    "read_file": ["open main.py", "show me the contents of app.tsx", "print requirements.txt"],
    "search_files": ["find all test files", "which files end with .tsx", "locate every yaml file"],
    "search_code": ["where do we call run_task", "find every place that uses sqlite", "grep for TODO"],
    "search_symbols": ["where is the Governor class defined", "go to definition of build_context", "which file declares LLMSettings"],
    "git_status": ["what have I modified", "any uncommitted work", "show changed files"],
    "git_diff": ["show the current diff", "what changed in router.py", "diff of my edits"],
    "git_log": ["last five commits", "who committed recently", "history of commits"],
    "run_tests": ["run the test suite", "execute pytest on tests/test_router.py", "check whether tests pass"],
    "lint": ["check for unused imports", "run the linter on backend", "any undefined names"],
    "syntax_check": ["is this python valid", "check syntax of calc.py", "does the module compile"],
    "system_status": ["how busy is the cpu", "memory pressure right now", "which process eats ram"],
    "memory_query": ["what was I working on yesterday", "remind me what we decided", "show unfinished work"],
    "project_overview": ["give me an overview of this codebase", "how is the repository organised", "what does this project do"],
    "explain_code": ["explain the retry logic in worker.py", "how does the context builder pick files", "what does select_ranges do"],
    "fix_bug": ["the add test is failing please fix it", "repair the crash in the parser", "tests broke after my change, fix them"],
    "analyze_error": ["why am I getting a KeyError here", "what causes this ImportError", "explain this stack trace"],
    "implement": ["add a --verbose flag to the cli", "implement caching for read_file", "create an endpoint that lists tasks"],
    "refactor": ["clean up the duplicated code in utils", "split this big function into smaller ones", "simplify the router chain logic"],
    "rename_symbol": ["rename function add to add_numbers", "rename class Foo to Bar everywhere"],
    "write_tests": ["add unit tests for the policy module", "write tests covering rollback", "generate pytest cases for calc"],
    "review": ["review my latest changes", "do a code review of the diff", "any problems in what I changed"],
    "create_file": ["create a new file utils/strings.py", "make a file named notes.md"],
    "general_coding": ["how do I read a csv in python", "what is the difference between a list and a tuple", "best way to structure a fastapi app"],
}


def _seed_hash() -> str:
    return hashlib.sha1(json.dumps(SEED, sort_keys=True).encode()).hexdigest()[:12]


class IntentModel:
    def __init__(self):
        self.pipe = None
        self.meta: dict = {}

    def load(self) -> bool:
        import joblib

        if not MODEL_PATH.exists() or not META_PATH.exists():
            return False
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            if meta.get("seed_hash") != _seed_hash():
                return False  # seed changed: the stored model is stale
            self.pipe, self.meta = joblib.load(MODEL_PATH), meta
            return True
        except Exception:
            return False

    def fit(self) -> dict:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import FeatureUnion, Pipeline

        texts = [t for k in SEED for t in SEED[k]]
        labels = [k for k in SEED for _ in SEED[k]]
        t0 = time.perf_counter()
        self.pipe = Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)),
            ])),
            ("clf", LogisticRegression(max_iter=2000, C=8.0, class_weight="balanced")),
        ])
        # Smallest class has 2 examples, so 2 folds is the honest maximum.
        cv = cross_val_score(self.pipe, texts, labels, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0))
        self.pipe.fit(texts, labels)
        self.meta = {"n_train": len(texts), "classes": len(SEED), "cv_folds": 2,
                     "cv_accuracy_mean": round(float(cv.mean()), 4), "cv_accuracy_std": round(float(cv.std()), 4),
                     "train_ms": round((time.perf_counter() - t0) * 1000, 1), "seed_hash": _seed_hash()}
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, MODEL_PATH)
        META_PATH.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return self.meta

    def predict(self, text: str) -> tuple[str, float, list[tuple[str, float]]]:
        if self.pipe is None and not self.load():
            self.fit()
        probs = self.pipe.predict_proba([text])[0]
        classes = list(self.pipe.classes_)
        order = probs.argsort()[::-1]
        return classes[order[0]], round(float(probs[order[0]]), 4), \
            [(classes[i], round(float(probs[i]), 4)) for i in order[:3]]


_model: IntentModel | None = None


def model() -> IntentModel:
    global _model
    if _model is None:
        _model = IntentModel()
        if not _model.load():
            _model.fit()
    return _model


# ------------------------------------------------------------------ parameters

# (?!\w) stops "config.json" from being read as "config.js".
_FILE_RX = re.compile(r"`?((?:[\w.-]+[/\\])*[\w.-]+\.(?:json|py|tsx|ts|jsx|js|md|csv|toml|yaml|yml|ini|cfg|txt|html|css|bat|ps1|sql))(?!\w)`?", re.I)
_DIR_RX = re.compile(r"\b(?:folder|directory|dir|in)\s+`?((?:[\w.-]+[/\\])*[\w.-]+)[/\\]?`?", re.I)
_GLOB_RX = re.compile(r"(\*\*?[\w./*-]*|[\w./-]*\*[\w./*-]*)")
_QUOTED_RX = re.compile(r"[\"'`]([^\"'`]{1,120})[\"'`]")
_LINES_RX = re.compile(r"\blines?\s+(\d+)(?:\s*(?:-|to|–|through)\s*(\d+))?", re.I)
_RENAME_RX = re.compile(r"\brename (?:the )?(?:function|class|method|variable|symbol|file)?\s*`?([\w./\\-]+)`?\s+to\s+`?([\w./\\-]+)`?", re.I)
_SYMBOL_HINT_RX = re.compile(r"\b(?:function|class|method|def|symbol|definition of|usages of|uses of|references to|calls to)\s+`?([A-Za-z_][\w.]*)`?", re.I)
_TEST_TARGET_RX = re.compile(r"((?:[\w.-]+/)*test[\w.-]*\.py(?:::[\w\[\]-]+)*)", re.I)
_N_RX = re.compile(r"\b(\d{1,3})\b")


def extract_params(text: str) -> dict:
    t = text or ""
    p: dict = {}
    files = [m.group(1).replace("\\", "/") for m in _FILE_RX.finditer(t)]
    if files:
        p["files"] = list(dict.fromkeys(files))
    d = _DIR_RX.search(t)
    if d and "." not in d.group(1).split("/")[-1] and d.group(1).lower() not in ("the", "this", "my", "a"):
        p["directory"] = d.group(1).replace("\\", "/")
    globs = [g for g in _GLOB_RX.findall(t) if "*" in g]
    if globs:
        p["glob"] = globs[0]
    q = _QUOTED_RX.findall(t)
    if q:
        p["quoted"] = q
    ln = _LINES_RX.search(t)
    if ln:
        p["start_line"] = int(ln.group(1))
        p["end_line"] = int(ln.group(2) or ln.group(1))
    rn = _RENAME_RX.search(t)
    if rn:
        p["rename_from"], p["rename_to"] = rn.group(1), rn.group(2)
    syms = [m.group(1) for m in _SYMBOL_HINT_RX.finditer(t)]
    idents = [w for w in re.findall(r"`([A-Za-z_][\w.]*)`|\b([A-Za-z_]+_[\w]+|[a-z]+[A-Z]\w*|[A-Z][a-z]+[A-Z]\w*)\b", t)]
    syms += [a or b for a, b in idents]
    syms = [s for s in dict.fromkeys(syms) if not any(s in f for f in files)]
    if syms:
        p["symbols"] = syms
    tt = _TEST_TARGET_RX.search(t)
    if tt:
        p["test_target"] = tt.group(1)
    n = _N_RX.search(t)
    if n:
        p["n"] = int(n.group(1))
    if re.search(r"Traceback \(most recent call last\)|^\s*File \".+\", line \d+|\w+Error:", t, re.M):
        p["has_traceback"] = True
    return p


def complexity(intent: str, text: str, params: dict) -> dict:
    """Transparent score -> low/medium/high. Drives model-tier choice; not a model."""
    base = {"implement": 3, "refactor": 3, "fix_bug": 2, "write_tests": 2, "rename_symbol": 1,
            "analyze_error": 1, "explain_code": 1, "review": 1, "create_file": 1, "general_coding": 1}.get(intent, 0)
    factors = {"intent": base}
    if len(params.get("files", [])) > 1 or re.search(r"\b(?:all|every|across|throughout|whole)\b", text, re.I):
        factors["multi_file"] = 1
    if params.get("has_traceback"):
        factors["error_present"] = 1
    if len(text) > 400:
        factors["long_request"] = 1
    if re.search(r"\b(?:architecture|design|migrat|concurren|thread|async|security|performance)\w*", text, re.I):
        factors["hard_topic"] = 1
    score = sum(factors.values())
    level = "low" if score <= 1 else ("medium" if score <= 3 else "high")
    return {"score": score, "level": level, "factors": factors}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    stage: str
    processor: str
    params: dict = field(default_factory=dict)
    complexity: dict = field(default_factory=dict)
    deterministic: bool = False
    alternatives: list = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def classify_intent(text: str, *, ml_min_conf: float | None = None) -> IntentResult:
    from backend.config import SETTINGS

    t0 = time.perf_counter()
    floor = SETTINGS.harness.ml_min_conf if ml_min_conf is None else ml_min_conf
    intent, conf, how = rule_intent(text)
    stage, alts = "rules", []
    if intent is None:
        try:
            intent, conf, alts = model().predict(text)
            stage, how = "ml", "coding_intent/tfidf_lr"
            if conf < floor:
                # Not confident enough to pick a specialised path: treat as a general
                # coding question (context + model), which is always safe.
                alts = [(intent, conf)] + [a for a in alts if a[0] != intent]
                intent, stage = "general_coding", "ml_low_confidence"
        except Exception as e:
            intent, conf, stage, how = "general_coding", 0.2, "fallback", f"fallback:{type(e).__name__}"
    params = extract_params(text)
    return IntentResult(intent=intent, confidence=round(conf, 4), stage=stage, processor=how, params=params,
                        complexity=complexity(intent, text, params), deterministic=intent in DETERMINISTIC,
                        alternatives=alts, latency_ms=round((time.perf_counter() - t0) * 1000, 3))
