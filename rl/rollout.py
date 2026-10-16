"""Stage 6 rollouts: multi-turn tool-using episodes against the current policy served by vLLM
(localhost:8000, model "m", deployment format), with verifiable rewards.

Tasks (rl/tasks.jsonl, built by rl/tasks.py):
  code    a KodCode problem; the agent works in an empty sandbox dir with tools, must leave solution.py;
          reward 1 if the problem's hidden pytest file passes, else 0.
  webdev  a MiMo webdev brief; the agent must leave a self-contained index.html (CDN libraries allowed);
          the page is rendered with Chromium (inside the MiMo webdev image via proot) and judged by a
          vision model against a fixed rubric; reward = score / 10, 0 if it doesn't render.
Every sandbox command runs as the unprivileged user `nobody` in its own directory, with a timeout.

usage: python rollout.py TASKS.jsonl OUT.jsonl N_TASKS GROUP SEED
Writes one line per rollout: {task_id, kind, group, reward, turns, messages, tools, ...}.
"""
import asyncio
import base64
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request

from openai import AsyncOpenAI

SB = os.environ.get("SANDBOX_ROOT", "/sandbox")
WEB_ROOTFS = os.environ.get("WEB_ROOTFS", "/workspace/fs/web")
PROOT = os.environ.get("PROOT", os.path.expanduser("~/.udocker/bin/proot-x86_64-4_8_0"))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")
MAX_TURNS, TURN_TOKENS, EPISODE_TOKENS = 12, 12288, 28000
SAMPLING = dict(temperature=1.0, top_p=0.95, extra_body={"top_k": 20})   # Qwen3.5 card, thinking mode

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a bash command in the working directory and return stdout+stderr (60 s timeout).",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file with the given content.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "submit", "description": "Finish the task. Call this once the deliverable is in place.",
     "parameters": {"type": "object", "properties": {}}}},
]
SYSTEM = ("You are a careful software engineer working in a sandboxed Linux directory. Use the tools to create "
          "and test files. Keep reasoning brief. Call submit when the deliverable is ready.")
CODE_TASK = ("{q}\n\nWrite the solution in `solution.py` in the working directory. The function must be "
             "`{decl}`. You may write and run your own tests with bash (python3 and pytest are available). "
             "Hidden tests will import it with `from solution import ...`.")
WEB_TASK = ("{q}\n\nDeliver a single self-contained `index.html` in the working directory. You may load libraries "
            "(e.g. React, Tailwind) from public CDNs, but there is no build step. It will be judged on a screenshot.")
RUBRIC = ("You are grading a website built from this brief:\n---\n{brief}\n---\nThe image is a full-page screenshot "
          "at 1280px wide. Score it 0-10 for how well it fulfils the brief as a polished, professional site: "
          "requested sections and content present, coherent layout and visual hierarchy, consistent styling that "
          "matches the requested style, readable text, no broken or overlapping elements. A blank or broken page is 0. "
          'Reply with JSON only: {{"score": <integer 0-10>, "reason": "<one sentence>"}}')


def chown(p):
    if os.geteuid() == 0:
        os.chown(p, 65534, 65534)


def _drop():  # run tool commands as the unprivileged user nobody (uid/gid 65534)
    os.setgroups([]); os.setgid(65534); os.setuid(65534)


def sh(cmd, cwd, timeout=60):
    # own session, so anything the command leaves running in the background (dev servers...) is killed with it
    # output goes to a file, and we wait for the command itself: a backgrounded server can't hold us up
    import tempfile
    with tempfile.TemporaryFile() as f:
        p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, preexec_fn=_drop if os.geteuid() == 0 else None,
                             start_new_session=True, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env={"PATH": os.environ.get("SANDBOX_PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": cwd})
        try:
            p.wait(timeout=timeout); note = ""
        except subprocess.TimeoutExpired:
            note = f"\n[timed out after {timeout}s]"
        finally:
            try:
                os.killpg(p.pid, 9)
            except ProcessLookupError:
                pass
            p.wait()
        f.seek(0); out = f.read().decode(errors="replace") + note
    return out[-4000:] if len(out) > 4000 else out


def safe_path(cwd, p):
    full = os.path.realpath(os.path.join(cwd, p))
    return full if full.startswith(os.path.realpath(cwd) + os.sep) or full == os.path.realpath(cwd) else None


def run_tool(name, args, cwd):
    if name == "bash":
        return sh(str(args.get("command", "")), cwd)
    if name in ("write_file", "read_file"):
        p = safe_path(cwd, str(args.get("path", "")))
        if not p:
            return "error: path must stay inside the working directory"
        if name == "read_file":
            try:
                return open(p).read()[-8000:]
            except Exception as e:
                return f"error: {e}"
        if not str(args.get("content", "")):
            return "error: write_file received no content (the call was probably cut off). Write a shorter file, or split it."
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(str(args.get("content", "")))
        chown(p)
        return f"wrote {os.path.relpath(p, cwd)} ({len(str(args.get('content', '')))} chars)"
    return f"error: unknown tool {name}"


def code_reward(task, cwd):
    if not os.path.exists(f"{cwd}/solution.py"):
        return 0.0, "no solution.py"
    d = f"{cwd}/.grade"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"{cwd}/solution.py", f"{d}/solution.py")
    open(f"{d}/test_hidden.py", "w").write(task["test"])
    chown(d)
    for f in os.listdir(d):
        chown(f"{d}/{f}")
    out = sh("python3 -m pytest -q -x -p no:cacheprovider test_hidden.py", d, timeout=60)
    return (1.0 if re.search(r"\b\d+ passed\b", out) and "failed" not in out and "error" not in out.lower() else 0.0), out[-300:]


def screenshot(cwd):
    shot = f"{cwd}/.shot.png"
    port = random.randint(20000, 60000)
    if not os.path.exists(f"{cwd}/index.html"):
        return None
    cmd = [PROOT, "-0", "-r", WEB_ROOTFS, "-b", "/proc", "-b", "/dev", "-b", "/sys", "-b", "/etc/resolv.conf",
           "-b", f"{cwd}:/workspace", "-w", "/workspace", "/usr/bin/env", "-i", "HOME=/root",
           "PATH=/usr/local/bin:/usr/bin:/bin", "PLAYWRIGHT_BROWSERS_PATH=/ms-playwright", "/bin/bash", "-c",
           # served over loopback HTTP: React/ES-module pages render blank from file:// (browsers block them)
           f"node -e \"require('http').createServer((q,s)=>require('fs').readFile('/workspace'+(q.url.split('?')[0]=='/'?'/index.html':q.url.split('?')[0]),(e,d)=>{{s.writeHead(e?404:200);s.end(d)}})).listen({port},'127.0.0.1')\" & "
           f"sleep 1; timeout 60 playwright screenshot --browser chromium --viewport-size=1280,800 --wait-for-timeout 3000 "
           f"--full-page http://127.0.0.1:{port}/index.html /workspace/.shot.png; kill %1"]
    try:
        subprocess.run(cmd, capture_output=True, timeout=90, env={**os.environ, "PROOT_NO_SECCOMP": "1"})
    except subprocess.TimeoutExpired:
        return None
    return shot if os.path.exists(shot) and os.path.getsize(shot) > 3000 else None


def judge(brief, png):
    key = os.environ.get("GEMINI_API_KEY")
    img = base64.b64encode(open(png, "rb").read()).decode()
    body = {"contents": [{"parts": [{"text": RUBRIC.format(brief=brief)}, {"inline_data": {"mime_type": "image/png", "data": img}}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent?key={key}"
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120))
            txt = r["candidates"][0]["content"]["parts"][-1]["text"]
            s = json.loads(txt)
            return max(0, min(10, int(s["score"]))) / 10, s.get("reason", "")[:200]
        except Exception as e:
            if attempt == 1:
                body["generationConfig"].pop("thinkingConfig", None)
            time.sleep(2 + 4 * attempt)
            err = repr(e)[:200]
    return None, "judge failed: " + err


async def episode(client, task, idx, k):
    cwd = f"{SB}/{task['task_id']}_{k}"
    shutil.rmtree(cwd, ignore_errors=True); os.makedirs(cwd); chown(cwd)
    prompt = (CODE_TASK.format(q=task["question"], decl=task["decl"]) if task["kind"] == "code"
              else WEB_TASK.format(q=task["question"]))
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    used, turns, submitted, finishes = 0, 0, False, []
    for turns in range(1, MAX_TURNS + 1):
        try:
            r = await client.chat.completions.create(model="m", messages=msgs, tools=TOOLS,
                                                     max_tokens=min(TURN_TOKENS, EPISODE_TOKENS - used), **SAMPLING)
        except Exception as e:
            return {"error": repr(e)[:300]}
        c = r.choices[0]
        used += r.usage.completion_tokens
        m = {"role": "assistant", "content": c.message.content or ""}
        finishes.append(c.finish_reason)
        rc = getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None)
        if rc:
            m["reasoning_content"] = rc
        calls = c.message.tool_calls or []
        if calls:
            m["tool_calls"] = [{"type": "function", "id": tc.id, "function": {"name": tc.function.name,
                                "arguments": tc.function.arguments or "{}"}} for tc in calls]   # API wants a JSON string
        msgs.append(m)
        if c.finish_reason == "length" or not calls:
            break
        for tc in m["tool_calls"]:
            if tc["function"]["name"] == "submit":
                submitted = True
                continue
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            out = await asyncio.to_thread(run_tool, tc["function"]["name"], args if isinstance(args, dict) else {}, cwd)
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
        if submitted or used >= EPISODE_TOKENS:
            break
    if task["kind"] == "code":
        reward, info = await asyncio.to_thread(code_reward, task, cwd)
    else:
        png = await asyncio.to_thread(screenshot, cwd)
        reward, info = (await asyncio.to_thread(judge, task["question"], png)) if png else (0.0, "no render")
    for m in msgs:  # the chat template (training) wants arguments as dicts
        for tc in m.get("tool_calls", []):
            try:
                a = json.loads(tc["function"]["arguments"])
            except Exception:
                a = {}
            tc["function"]["arguments"] = a if isinstance(a, dict) else {}
    res = {"task_id": task["task_id"], "kind": task["kind"], "group": idx, "reward": reward, "info": info,
           "turns": turns, "tokens": used, "submitted": submitted, "finishes": finishes, "messages": msgs, "tools": TOOLS}
    shutil.rmtree(cwd, ignore_errors=True)
    return res


async def main():
    TASKS, OUT, N_TASKS, GROUP, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    tasks = [json.loads(l) for l in open(TASKS)]
    rng = random.Random(SEED)
    pick = rng.sample(tasks, N_TASKS)
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x", timeout=1800)
    sem = asyncio.Semaphore(int(os.environ.get("CONCURRENCY", 64)))

    async def one(t, i, k):
        async with sem:
            try:
                return await episode(client, t, i, k)
            except Exception as e:  # one broken episode must not sink the batch
                return {"task_id": t["task_id"], "kind": t["kind"], "group": i, "error": repr(e)[:300]}

    res = await asyncio.gather(*(one(t, i, k) for i, t in enumerate(pick) for k in range(GROUP)))
    with open(OUT, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in res if "error" not in r]
    for kind in ("code", "webdev"):
        rs = [r["reward"] for r in ok if r["kind"] == kind and r["reward"] is not None]
        if rs:
            print(f"{kind}: n={len(rs)} mean_reward={sum(rs)/len(rs):.3f}", flush=True)
    print(f"errors={len(res)-len(ok)} mean_turns={sum(r['turns'] for r in ok)/max(1,len(ok)):.1f} "
          f"mean_tokens={sum(r['tokens'] for r in ok)/max(1,len(ok)):.0f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
