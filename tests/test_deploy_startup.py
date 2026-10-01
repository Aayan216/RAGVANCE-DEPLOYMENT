import os
import subprocess
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, r"D:\v5\RAG-p2")
os.chdir(r"D:\v5\RAG-p2")

PASSED = 0
FAILED = 0


def ok(label, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print("PASS:", label, flush=True)
    else:
        FAILED += 1
        print("FAIL:", label, ("| " + str(extra)) if extra != "" else "", flush=True)


CHILD_BOOT = r"""
import os, sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()
import backend.views as views

BANNED = ["torch", "sentence_transformers", "fitz", "docx", "pptx", "pandas",
          "sklearn", "langchain_google_genai", "mock_test"]
loaded = sorted(m for m in BANNED if m in sys.modules)
print("BANNED_LOADED=" + ",".join(loaded), flush=True)

SINGLETONS = ["vector_store", "embedder", "rag_chain", "mcq_generator",
              "mock_test_service", "analyzer"]
not_proxy = [n for n in SINGLETONS
             if not isinstance(getattr(views, n), views._LazyService)]
print("NOT_PROXY=" + ",".join(not_proxy), flush=True)
built = [n for n in SINGLETONS if getattr(views, n)._obj is not None]
print("ALREADY_BUILT=" + ",".join(built), flush=True)

rss = views._rss_mb()
print("RSS=" + (f"{rss:.1f}" if rss is not None else "None"), flush=True)

calls = []
class Thing:
    def greet(self, x):
        return "hi-" + x
p = views._LazyService("probe", lambda: (calls.append(1), Thing())[1])
a = p.greet("a")
b = p.greet("b")
print("ATTR_OK=" + str(a == "hi-a" and b == "hi-b"), flush=True)
print("BUILD_ONCE=" + str(len(calls) == 1), flush=True)

empty = views._LazyService("empty", lambda: [])
print("BOOL_TRUE=" + str(bool(empty) is True), flush=True)
print("LEN_OK=" + str(len(empty) == 0 and len(calls) == 1), flush=True)

dunder_calls = []
d = views._LazyService("dunder", lambda: (dunder_calls.append(1), Thing())[1])
try:
    d.__wrapped__
    dunder_blocked = False
except AttributeError:
    dunder_blocked = True
print("DUNDER_BLOCKED=" + str(dunder_blocked and not dunder_calls), flush=True)

c = views._LazyService("callable", lambda: (lambda *args: len(args)))
print("CALL_OK=" + str(c(1, 2, 3) == 3), flush=True)

class Box:
    pass
b = views._LazyService("box", lambda: Box())
b.value = 7
print("SETATTR_OK=" + str(b.value == 7 and b._obj.value == 7), flush=True)
print("CHILD_DONE", flush=True)
"""

child = subprocess.run(
    [sys.executable, "-c", CHILD_BOOT],
    cwd=r"D:\v5\RAG-p2",
    capture_output=True,
    text=True,
    timeout=300,
)
out = child.stdout + child.stderr

ok("boot subprocess exited 0", child.returncode == 0, out[-2000:])

banned_line = next((l for l in out.splitlines() if l.startswith("BANNED_LOADED=")), None)
banned = banned_line.split("=", 1)[1] if banned_line else "<missing>"
ok("no heavy modules imported at boot", banned_line is not None and banned == "",
   banned or out[-1500:])

not_proxy_line = next((l for l in out.splitlines() if l.startswith("NOT_PROXY=")), None)
not_proxy = not_proxy_line.split("=", 1)[1] if not_proxy_line else "<missing>"
ok("all 6 services are lazy proxies", not_proxy_line is not None and not_proxy == "",
   not_proxy)

built_line = next((l for l in out.splitlines() if l.startswith("ALREADY_BUILT=")), None)
built = built_line.split("=", 1)[1] if built_line else "<missing>"
ok("no service built during boot", built_line is not None and built == "", built)

ok("[MEM] rss= logged at import (Render log line)",
   "[MEM] rss=" in out, out[-800:])

rss_line = next((l for l in out.splitlines() if l.startswith("RSS=")), None)
rss_val = None
if rss_line and rss_line != "RSS=None":
    try:
        rss_val = float(rss_line.split("=", 1)[1])
    except ValueError:
        rss_val = None
ok("boot RSS measurable", rss_val is not None, rss_line)
ok("boot RSS under 300MB", rss_val is not None and rss_val < 300, rss_line)


def child_flag(name):
    line = next((l for l in out.splitlines() if l.startswith(name + "=")), None)
    return line.split("=", 1)[1] if line else None


ok("proxy attribute pass-through", child_flag("ATTR_OK") == "True", child_flag("ATTR_OK"))
ok("proxy builds exactly once", child_flag("BUILD_ONCE") == "True", child_flag("BUILD_ONCE"))
ok("proxy truthy even when underlying falsy", child_flag("BOOL_TRUE") == "True",
   child_flag("BOOL_TRUE"))
ok("proxy len() pass-through", child_flag("LEN_OK") == "True", child_flag("LEN_OK"))
ok("dunder probe blocked without building", child_flag("DUNDER_BLOCKED") == "True",
   child_flag("DUNDER_BLOCKED"))
ok("proxy call pass-through", child_flag("CALL_OK") == "True", child_flag("CALL_OK"))
ok("proxy setattr routes to built service", child_flag("SETATTR_OK") == "True",
   child_flag("SETATTR_OK"))
ok("child completed", "CHILD_DONE" in out, out[-1500:])

yaml_src = open(r"D:\v5\RAG-p2\render.yaml", encoding="utf-8").read()
ok("render.yaml bakes model into project dir",
   "HF_HOME=/opt/render/project/src/.hf_cache" in yaml_src)
ok("render.yaml sets runtime HF_HOME", "- key: HF_HOME" in yaml_src)
ok("render.yaml sets runtime HF_HUB_OFFLINE", '- key: HF_HUB_OFFLINE' in yaml_src)

check = subprocess.run(
    [sys.executable, "manage.py", "check"],
    cwd=r"D:\v5\RAG-p2", capture_output=True, text=True, timeout=300,
)
ok("manage.py check clean", check.returncode == 0,
   (check.stdout + check.stderr)[-1500:])

mig = subprocess.run(
    [sys.executable, "manage.py", "makemigrations", "--check", "--dry-run"],
    cwd=r"D:\v5\RAG-p2", capture_output=True, text=True, timeout=300,
)
ok("makemigrations --check clean", mig.returncode == 0,
   (mig.stdout + mig.stderr)[-1500:])

print(f"\n===== {PASSED}/{PASSED + FAILED} passed =====", flush=True)
sys.exit(1 if FAILED else 0)
