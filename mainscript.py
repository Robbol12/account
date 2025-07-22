import os
import re
import json
import string
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from llama_cpp import Llama
from nltk.tokenize import sent_tokenize
import time
import sys
import glob


from openai import OpenAI


# NOTE: Replace with your actual valid OpenAI API key
client = OpenAI(api_key="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")


MODEL_PATH = "/workspace/models/Mixtral-8x7B-Instruct-v0.1.Q6_K.gguf"


N_THREADS = 16
N_CTX = 3072
N_GPU_LAYERS = 30
CHUNK_BASE_DIR = "chunks"
TRANSCRIPT_SAVE_DIR = "checks"
SENTENCE_CAP = 50
BATCH_SIZE = 10


llama_model = None


def load_model():
    global llama_model
    try:
        print("[...] Loading Mistral model...")


        llama_model = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            chat_format="mistral-instruct",  # ✅ REQUIRED for Mixtral instruct models
            verbose=False
        )


        print("[✓] Mistral model loaded successfully.")
    except Exception as e:
        print(f"[!] Could not load Mistral model: {e}")
        llama_model = None


def extract_transcript_targets(question):
    system_prompt = (
        "You're helping fetch earnings call transcripts based on a user's question. "
        "From the question, extract:\n"
        "1) The company's ticker (or name),\n"
        "2) The quarter and year where the claim was made (if any),\n"
        "3) A list of [quarter, year] pairs the script should pull (either the whole year, or a range).\n\n"
        "Rules:\n"
        "- If the user says 'in 2022', return Q1-Q4 2022.\n"
        "- If they say 'Q1 2022', return that quarter and all following ones that could show if it happened.\n"
        "- If they say 'from 2021 to 2023', return all quarters in that range.\n"
        "- If they say 'last year', assume it's 2024.\n\n"
        "Respond ONLY in this JSON format:\n"
        "{\"ticker\": \"VIST\", \"claim_quarter\": null, \"claim_year\": 2022, \"check_periods\": [[1,2022],[2,2022],[3,2022],[4,2022]]}"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[!] Failed to parse GPT response: {e}")
        return None


def safe_label(label):
    return ''.join(c for c in label if c in string.ascii_letters + string.digits + "_")


def process_batch(start_idx, sentences, question, chunk_dir):
    BATCH_SIZE = len(sentences)
    results = []


    all_cached = all(
        os.path.exists(os.path.join(chunk_dir, f"{start_idx + i:03d}_response.txt"))
        for i in range(BATCH_SIZE)
    )


    if all_cached:
        for i in range(BATCH_SIZE):
            with open(os.path.join(chunk_dir, f"{start_idx + i:03d}_response.txt"), "r", encoding="utf-8") as f:
                cached = f.read().strip()
            results.append((start_idx + i, cached, True))
        print(f"[✓] Loaded batch {start_idx}–{start_idx + BATCH_SIZE - 1} from cache")
        return results


    for i, sentence in enumerate(sentences):
        sentence_path = os.path.join(chunk_dir, f"{start_idx + i:03d}.txt")
        if not os.path.exists(sentence_path):
            with open(sentence_path, "w", encoding="utf-8") as f:
                f.write(sentence)


    numbered = [f"Sentence {start_idx + i + 1}: {s}" for i, s in enumerate(sentences)]


    prompt = f"""[INST] Question: {question}
Determine if each sentence helps answer the question.
Reply with a brief answer (max 15 words) or "No info found".


{chr(10).join(numbered)}
[/INST]
"""


    print(f"\n[⇡] Batching {BATCH_SIZE} chunks starting at {start_idx}...")


    try:
        start = time.time()
        response = llama_model(prompt, temperature=0.2, max_tokens=256)
        end = time.time()
        raw_output = response["choices"][0]["text"].strip()


        lines = raw_output.split("\n")
        for i, line in enumerate(lines[:BATCH_SIZE]):
            result = line.strip() or "No info found."
            idx = start_idx + i
            results.append((idx, result, False))


            with open(os.path.join(chunk_dir, f"{idx:03d}_response.txt"), "w", encoding="utf-8") as f:
                f.write(result)


        print(f"[✓] Batch processed in {end - start:.2f}s")
    except Exception as e:
        print(f"[!] Batch {start_idx} failed: {e}")
        for i in range(BATCH_SIZE):
            results.append((start_idx + i, "No info found.", False))


    return results


def summarize_final(question, excerpts):
    joined_excerpts = "\n---\n".join(excerpts)
    prompt = f"""[INST]
The user asked:


"{question}"


Here are relevant excerpts from the transcript:
{joined_excerpts}


Based on this, did the company follow through on the claim? Respond ONLY in this JSON format:
{{
  "follow_through": true or false,
  "evidence": "Brief quote or summary from the excerpts above"
}}
[/INST]
"""
    try:
        start = time.time()
        response = llama_model(prompt, temperature=0.3, max_tokens=512)
        end = time.time()
        raw = response["choices"][0]["text"].strip()
        print(f"\n[MISTRAL FINAL ANSWER] ({end - start:.2f}s):\n{raw}")


        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0)), raw
    except Exception as e:
        print(f"[!] Final JSON parse failed: {e}")


    return {
        "follow_through": False,
        "evidence": "No evidence found in available text."
    }, ""



def summarize_transcript(quarter_text, question, quarter_label):
    sentences = sent_tokenize(quarter_text)[:SENTENCE_CAP]
    question_safe_label = safe_label(question)[:60]
    safe_quarter_label = safe_label(quarter_label)
    chunk_dir = os.path.join(CHUNK_BASE_DIR, question_safe_label, safe_quarter_label)
    os.makedirs(chunk_dir, exist_ok=True)


    summaries, evidence_links, seen_responses = [], [], set()
    total_cache_hits, i = 0, 0
    start_time = time.time()


    while i < len(sentences):
        batch = sentences[i:i + BATCH_SIZE]
        print(f"\n{safe_quarter_label} — Batch #{i // BATCH_SIZE + 1} → Chunks {i}–{i + len(batch) - 1}")
        results = process_batch(i, batch, question, chunk_dir)


        for b_offset, (idx, result, from_cache) in enumerate(results):
            if from_cache:
                total_cache_hits += 1


            if "no info" in result.lower():
                continue


            if result not in seen_responses:
                seen_responses.add(result)
                summaries.append(result)
                # Always map using your own code's sentence index and transcript text
                evidence_links.append({
                    "sentence_number": idx + 1,
                    "sentence": sentences[idx],
                    "response": result
                })


        i += BATCH_SIZE


    print(f"\n[✓] {total_cache_hits}/{len(sentences)} chunks loaded from cache.")


    evidence_links.sort(key=lambda x: x['sentence_number'])


    if summaries:
        final, raw_final_output = summarize_final(question, summaries)
    else:
        final, raw_final_output = {
            "follow_through": False,
            "evidence": "No evidence found in available text."
        }, ""


    meta = {
        "quarter": safe_quarter_label,
        "question": question,
        "sentence_count": len(sentences),
        "cache_hits": total_cache_hits,
        "run_time_sec": round(time.time() - start_time, 2),
        "final_answer": final,
        "excerpts_used": summaries,
        "raw_final_output": raw_final_output,
        "evidence_links": evidence_links
    }
    results_dir = "results/" + safe_label(question)[:60]  # always create per-question results directory
    os.makedirs(results_dir, exist_ok=True)
    meta_path = os.path.join(results_dir, f"{safe_quarter_label}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


    print(f"[✓] Final result written to: {meta_path}")
    print(f"\n⏱️ Total summarization time: {meta['run_time_sec']} seconds")
    return final


def get_or_load_transcript(ticker, quarter, year):
    filename = f"{ticker.upper()}_Q{quarter}_{year}.txt"
    filepath = os.path.join(TRANSCRIPT_SAVE_DIR, filename)


    if os.path.exists(filepath):
        choice = input(f"\n[✓] Found saved transcript for Q{quarter} {year}. Use it? (Y/N): ").strip().upper()
        if choice == "Y":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()


    print(f"\nPaste transcript for Q{quarter} {year}. Type '--- END ---' on its own line when finished:\n")
    text = get_multiline_input()


    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


    return text


def get_multiline_input():
    print("Paste the full transcript below, then press:")
    print("  - Ctrl+D (Mac/Linux) or Ctrl+Z then Enter (Windows) to finish.\n")
    return sys.stdin.read().strip()


def prompt_transcript_usage(transcript_list):
    print("\n[✓] Found saved transcripts for all requested periods:")
    for idx, (label, path) in enumerate(transcript_list, 1):
        print(f"  {idx}. {label}")


    choice = input("\nUse all saved transcripts? (Y/N): ").strip().upper()
    if choice == "Y":
        use_flags = [True] * len(transcript_list)
    else:
        print(
            "\nWhich transcript(s) do you want to change?"
            " Enter indices separated by space (e.g., 1 3),"
            " or 0 to change all:"
        )
        resp = input("> ").strip()
        if resp == "0":
            use_flags = [False] * len(transcript_list)
        else:
            set_to_replace = set(int(i)-1 for i in resp.split() if i.isdigit())
            use_flags = [i not in set_to_replace for i in range(len(transcript_list))]
    return use_flags


def quarter_sort_key(filename):
    match = re.search(r'Q(\d+)_(\d+)_meta\.json', filename)
    if match:
        q, y = int(match.group(1)), int(match.group(2))
        return (y, q)
    return (9999, 0)  # place malformed names at end


if __name__ == "__main__":
    import nltk
    nltk.download("punkt")


    os.makedirs(CHUNK_BASE_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_SAVE_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)


    load_model()
    if llama_model is None:
        exit(1)


    user_question = input("What's your accountability question? ").strip()
    question_safe_label = safe_label(user_question)[:60]
    info = extract_transcript_targets(user_question)


    if not info:
        print("[x] Could not extract metadata from question.")
        exit(1)


    ticker = info["ticker"]
    claim_q = info["claim_quarter"]
    claim_y = info["claim_year"]
    check_periods = info["check_periods"]


    all_blocks = []
    seen = set()


    transcript_periods = []
    period_keys = []
    if claim_q and claim_y:
        seen.add((claim_q, claim_y))
        transcript_periods.append((f"Q{claim_q} {claim_y}", os.path.join(TRANSCRIPT_SAVE_DIR, f"{ticker.upper()}_Q{claim_q}_{claim_y}.txt")))
        period_keys.append((claim_q, claim_y))
    for q, y in check_periods:
        if (q, y) in seen:
            continue
        transcript_periods.append((f"Q{q} {y}", os.path.join(TRANSCRIPT_SAVE_DIR, f"{ticker.upper()}_Q{q}_{y}.txt")))
        period_keys.append((q, y))


    all_exist = all(os.path.exists(path) for _, path in transcript_periods)
    if all_exist and transcript_periods:
        use_saved = prompt_transcript_usage(transcript_periods)
    else:
        use_saved = [False] * len(transcript_periods)


    all_blocks = []
    for i, ((label, path), use_it) in enumerate(zip(transcript_periods, use_saved)):
        q, y = period_keys[i]
        if use_it and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = get_or_load_transcript(ticker, q, y)
        all_blocks.append((label, text))


    if not all_blocks:
        print("[x] No transcripts provided.")
        exit(1)


    merged_path = os.path.join(TRANSCRIPT_SAVE_DIR, f"{ticker.upper()}_claim_followup.txt")
    with open(merged_path, "w", encoding="utf-8") as f:
        for label, txt in all_blocks:
            f.write(f"--- {label} ---\n{txt.strip()}\n\n")
    print(f"\n[✓] Transcripts saved to {merged_path}")


    results_root = os.path.join("results", question_safe_label)
    os.makedirs(results_root, exist_ok=True)


    for label, text in all_blocks:
        print(f"\n===== Analyzing {label} =====")
        result = summarize_transcript(text.strip(), user_question, label)
        print(f"[SUMMARY for {label}]:\n{json.dumps(result, indent=2)}")


    print("\n[⏳] Synthesizing all meta-results for final GPT-3.5 turbo review...")


    meta_files = sorted(glob.glob(os.path.join(results_root, "*_meta.json")), key=quarter_sort_key)
    all_quarters = []
    for mf in meta_files:
        with open(mf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_quarters.append(data)


    prompt_parts = [f"Question: {user_question}\n"]
    for qdat in all_quarters:
        prompt_parts.append(
            f"For {qdat['quarter']}:\n"
            f"    Final answer: {qdat['final_answer']}\n"
            f"    Evidence: {qdat['final_answer']['evidence']}\n"
            f"    Key Sentences:\n"
        )
        for ev in qdat.get("evidence_links", []):
            prompt_parts.append(f"        - {ev['sentence']}")


    prompt_parts.append(
        "\nBased on all the above quarterly evidence, synthesize a single summary answer: "
        "Did the company follow through on the claim? Please explain your reasoning based on the detailed evidence, "
        "highlighting trends, contradictions, and clear verdict. Be specific. Return a short summary first, "
        "then a short explanation in bullet points."
    )
    prompt_for_gpt = "\n".join(prompt_parts)


    # FIXED: Changed client.Chat.Completions.create to client.chat.completions.create
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial transcript analyst. Given the evidence and quarterly analysis below, provide a synthesized final answer to the user's question. First give a clear Yes/No/Uncertain verdict, then briefly explain using the supplied evidence."},
            {"role": "user", "content": prompt_for_gpt}
        ],
        max_tokens=600
    )
    final_summary = response.choices[0].message.content


    print("\n===== OVERALL FINAL JUDGMENT =====\n")
    print(final_summary)
    final_outfile = os.path.join(results_root, "overall_final_judgment.txt")
    with open(final_outfile, "w", encoding="utf-8") as f:
        f.write(final_summary)
    print(f"[✓] Overall final judgment written to: {final_outfile}")