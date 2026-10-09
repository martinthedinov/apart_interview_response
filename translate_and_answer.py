# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.40"]
# ///
"""Multilingual probe: translate each voter question into the election's local language, then ask a
web-search-enabled system-under-test the same question in English and in the local language.

Writes answers in the same schema as results/answers.jsonl so score.py can judge them:
  data/questions_native.jsonl            questions with `prompt` replaced by the translation (+ prompt_en, lang)
  results/answers_<tag>_en.jsonl         SUT answers to the English prompts
  results/answers_<tag>_native.jsonl     SUT answers to the translated prompts

Env: OPENROUTER_API_KEY. TRANSLATOR (default openai/gpt-4o-mini), SUT (default openai/gpt-4o-mini:online).
Run: uv run translate_and_answer.py
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent
LANG = {"de-st-2026": "German", "fi-2027": "Finnish", "br-2026": "Brazilian Portuguese"}
TRANSLATOR = os.environ.get("TRANSLATOR", "openai/gpt-4o-mini")
SUT = os.environ.get("SUT", "openai/gpt-4o-mini:online")
TAG = SUT.split("/")[-1].replace(":", "-")

client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict]):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def translate(q: dict) -> dict:
    lang = LANG[q["election_id"]]
    r = client.chat.completions.create(model=TRANSLATOR, temperature=0, messages=[{"role": "user", "content":
        f"Translate this voter's question into natural {lang}, as a local voter would ask it. Keep all dates, "
        f"names and meaning exactly; do not add or drop information. Output ONLY the translation.\n\n{q['prompt']}"}])
    return q | {"prompt_en": q["prompt"], "lang": lang, "prompt": r.choices[0].message.content.strip()}


def answer(q: dict) -> dict:
    row = {k: q[k] for k in ("qid", "election_id", "field_key", "risk_tier", "prompt")} | {
        "model": TAG, "lang": q.get("lang", "English")}
    try:
        r = client.chat.completions.create(model=SUT, messages=[{"role": "user", "content": q["prompt"]}])
        msg = r.model_dump()["choices"][0]["message"]
        cites = [a["url_citation"]["url"] for a in (msg.get("annotations") or []) if a.get("type") == "url_citation"]
        return row | {"answer_text": msg["content"], "citations": cites, "search_queries": [],
                      "status": "completed", "model_returned": r.model, "error": None}
    except Exception as e:  # noqa: BLE001 -- record and keep going; score.py sees status/error
        return row | {"answer_text": "", "citations": [], "search_queries": [],
                      "status": "failed", "model_returned": None, "error": f"{type(e).__name__}: {e}"}


def main():
    # NATIVE_IN=<file> SUFFIX=<name>: skip translation, just answer an existing translated question set
    if os.environ.get("NATIVE_IN"):
        with ThreadPoolExecutor(8) as ex:
            ans = list(ex.map(answer, load_jsonl(Path(os.environ["NATIVE_IN"]))))
        write_jsonl(HERE / "results" / f"answers_{TAG}_{os.environ.get('SUFFIX', 'custom')}.jsonl", ans)
        print(f"answered {len(ans)}, failed: {[a['qid'] for a in ans if a['status'] != 'completed'] or 'none'}")
        return
    questions = load_jsonl(HERE / "data" / "questions.jsonl")
    with ThreadPoolExecutor(8) as ex:
        native = list(ex.map(translate, questions))
        write_jsonl(HERE / "data" / "questions_native.jsonl", native)
        en_ans = list(ex.map(answer, questions))
        nat_ans = list(ex.map(answer, native))
    write_jsonl(HERE / "results" / f"answers_{TAG}_en.jsonl", en_ans)
    write_jsonl(HERE / "results" / f"answers_{TAG}_native.jsonl", nat_ans)
    failed = [a["qid"] for a in en_ans + nat_ans if a["status"] != "completed"]
    print(f"translated {len(native)}, answered {len(en_ans) + len(nat_ans)}, failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
