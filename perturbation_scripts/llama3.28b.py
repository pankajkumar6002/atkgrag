#!/usr/bin/env python3

import argparse
import copy
import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from bert_score import score as bert_score
from SPARQLWrapper import JSON, SPARQLWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer


LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
QWEN_DEFAULT_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3.5-9B")
DEFAULT_SPARQL_ENDPOINT = "http://localhost:3002/sparql"


@dataclass(frozen=True)
class PromptSpec:
	set_id: int
	name: str
	system: str
	user: str


PROMPTS: Sequence[PromptSpec] = [
	PromptSpec(
		set_id=1,
		name="synonym_alias",
		system=(
			"You are an adversarial question rewriter for KGQA robustness. "
			"Rewrite exactly one question by replacing only content words with synonyms or Freebase aliases. "
			"Keep grammar, topic entity, predicate, and denotation unchanged. "
			"Return one single-line question and nothing else."
		),
		user="Question: {question}\nCanonical answer hints: {answer_hint}\nRewrite using synonyms/aliases only.",
	),
	PromptSpec(
		set_id=2,
		name="structural",
		system=(
			"You are an adversarial question rewriter. "
			"Apply one syntactic transformation while preserving topic entity, predicate, temporal/numeric constraints, and answer denotation. "
			"Return one single-line question only."
		),
		user="Question: {question}\nCanonical answer hints: {answer_hint}\nApply one structural transformation.",
	),
	PromptSpec(
		set_id=3,
		name="semantic_drift",
		system=(
			"Rewrite with high lexical drift but unchanged meaning. "
			"Do not add or remove facts. Preserve denotation exactly. "
			"Return one single-line question only."
		),
		user="Question: {question}\nCanonical answer hints: {answer_hint}\nMaximize lexical drift with meaning preserved.",
	),
	PromptSpec(
		set_id=4,
		name="compositional",
		system=(
			"Apply exactly one KG-safe transformation: hop-order phrasing shift, true constraint injection, or alias substitution. "
			"Forbidden: predicate change, scope expansion, answer change. "
			"Return one single-line question only."
		),
		user="Question: {question}\nCanonical answer hints: {answer_hint}\nApply one compositional-safe transformation.",
	),
	PromptSpec(
		set_id=5,
		name="reorder",
		system=(
			"Reorder phrasing while keeping semantics and denotation identical. "
			"Keep entities, constraints, and predicate unchanged. "
			"Return one single-line question only."
		),
		user="Question: {question}\nCanonical answer hints: {answer_hint}\nReorder wording only.",
	),
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate 5 perturbation sets for CWQ and WebQSP with Llama and constraint filtering.")
	parser.add_argument("--cwq-dir", required=True, type=Path, help="Path to CWQ_ori directory")
	parser.add_argument("--webqsp-dir", required=True, type=Path, help="Path to webqsp_ori directory")
	parser.add_argument("--out-root", required=True, type=Path, help="Output directory under tmlr/llama_perturbation_sets")
	parser.add_argument("--llama-model", default=LLAMA_MODEL_ID, help="Llama model ID/path")
	parser.add_argument("--qwen-model", default=QWEN_DEFAULT_PATH, help="Qwen model path for A* SPARQL generation")
	parser.add_argument("--sparql-endpoint", default=DEFAULT_SPARQL_ENDPOINT, help="SPARQL endpoint URL")
	parser.add_argument("--batch-size", default=8, type=int)
	parser.add_argument("--max-new-tokens", default=64, type=int)
	parser.add_argument("--attempts", default=4, type=int)
	parser.add_argument("--smoke-size", default=20, type=int, help="Samples per file for smoke run; use <=0 for full")
	parser.add_argument("--ppl-threshold", default=50.0, type=float)
	parser.add_argument("--bertscore-threshold", default=0.85, type=float)
	parser.add_argument("--gpu-list", default="0,1,2", help="Visible GPUs list. GPU 3 is intentionally excluded by default.")
	return parser.parse_args()


def clean_text(text: str) -> str:
	text = text.strip().replace("\n", " ").replace("\t", " ")
	text = re.sub(r"\s+", " ", text)
	text = re.sub(r"^(question|rewritten question|output)\s*:\s*", "", text, flags=re.IGNORECASE)
	return text.strip(" \"'")


def load_chat_model(model_name_or_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
	try:
		tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
	except Exception:
		tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(
		model_name_or_path,
		torch_dtype=torch.bfloat16,
		device_map="auto",
		low_cpu_mem_usage=True,
	)
	model.eval()
	return tokenizer, model


def batched_generate(
	tokenizer: AutoTokenizer,
	model: AutoModelForCausalLM,
	batch_messages: Sequence[List[Dict[str, str]]],
	max_new_tokens: int,
) -> List[str]:
	prompts = [
		tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
		for msgs in batch_messages
	]
	encoded = tokenizer(
		prompts,
		return_tensors="pt",
		padding=True,
		truncation=True,
	).to(model.device)

	with torch.inference_mode():
		outputs = model.generate(
			**encoded,
			max_new_tokens=max_new_tokens,
			do_sample=True,
			temperature=0.7,
			top_p=0.9,
			eos_token_id=tokenizer.eos_token_id,
			pad_token_id=tokenizer.pad_token_id,
		)

	input_len = encoded["input_ids"].shape[1]
	decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
	return [clean_text(x) for x in decoded]


def compute_ppl(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, text: str) -> float:
	if not text:
		return float("inf")
	enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
	with torch.inference_mode():
		out = model(**enc, labels=enc["input_ids"])
	return float(torch.exp(out.loss).detach().cpu())


def run_sparql(endpoint: str, query: str) -> Optional[set]:
	try:
		client = SPARQLWrapper(endpoint)
		client.setQuery(query)
		client.setReturnFormat(JSON)
		result = client.query().convert()
		bindings = result.get("results", {}).get("bindings", [])
		normalized = set()
		for row in bindings:
			parts = []
			for key in sorted(row):
				parts.append(f"{key}={row[key].get('value', '')}")
			normalized.add(" | ".join(parts))
		return normalized
	except Exception:
		return None


def build_answer_hint(record: Dict[str, Any], dataset_name: str) -> str:
	if dataset_name == "cwq":
		answers = [x.get("answer", "") for x in record.get("answers", []) if isinstance(x, dict)]
		return "; ".join([x for x in answers if x]) or "unknown"

	parses = record.get("Parses", [])
	if not parses:
		return "unknown"
	answers = parses[0].get("Answers", [])
	names = [x.get("EntityName", "") for x in answers if isinstance(x, dict)]
	return "; ".join([x for x in names if x]) or "unknown"


def extract_question(record: Dict[str, Any], dataset_name: str) -> str:
	if dataset_name == "cwq":
		return str(record.get("question", "")).strip()
	return str(record.get("ProcessedQuestion") or record.get("RawQuestion") or "").strip()


def extract_gold_sparql(record: Dict[str, Any], dataset_name: str) -> Optional[str]:
	if dataset_name == "cwq":
		return record.get("sparql")
	parses = record.get("Parses", [])
	if not parses:
		return None
	return parses[0].get("Sparql")


def generate_sparql_with_qwen(
	qwen_tok: AutoTokenizer,
	qwen_model: AutoModelForCausalLM,
	original_q: str,
	perturbed_q: str,
	gold_sparql: str,
) -> Optional[str]:
	messages = [
		{
			"role": "system",
			"content": (
				"You generate Freebase SPARQL for a paraphrased question. "
				"Preserve denotation and output only SPARQL text."
			),
		},
		{
			"role": "user",
			"content": (
				f"Original question: {original_q}\n"
				f"Original gold SPARQL:\n{gold_sparql}\n\n"
				f"Paraphrased question: {perturbed_q}\n"
				"Return SPARQL only."
			),
		},
	]
	out = batched_generate(qwen_tok, qwen_model, [messages], max_new_tokens=220)[0]
	out = out.strip()
	if "SELECT" not in out.upper() and "ASK" not in out.upper():
		return None
	return out


def evaluate_candidate(
	original_question: str,
	candidate_question: str,
	gold_sparql: Optional[str],
	llama_tok: AutoTokenizer,
	llama_model: AutoModelForCausalLM,
	qwen_tok: AutoTokenizer,
	qwen_model: AutoModelForCausalLM,
	sparql_endpoint: str,
	ppl_threshold: float,
	bert_threshold: float,
) -> Tuple[bool, Dict[str, Any]]:
	if not candidate_question:
		return False, {"reason": "empty_generation"}

	ppl = compute_ppl(llama_tok, llama_model, candidate_question)
	P, R, F1 = bert_score(
		[candidate_question],
		[original_question],
		lang="en",
		model_type="roberta-large",
		verbose=False,
		device=str(llama_model.device),
	)
	bert_f1 = float(F1[0].detach().cpu())

	if ppl >= ppl_threshold:
		return False, {"ppl": ppl, "bertscore": bert_f1, "reason": "ppl_threshold"}
	if bert_f1 <= bert_threshold:
		return False, {"ppl": ppl, "bertscore": bert_f1, "reason": "bertscore_threshold"}

	if not gold_sparql:
		return False, {"ppl": ppl, "bertscore": bert_f1, "reason": "missing_gold_sparql"}

	original_answers = run_sparql(sparql_endpoint, gold_sparql)
	if original_answers is None:
		return False, {"ppl": ppl, "bertscore": bert_f1, "reason": "gold_sparql_failed"}

	pred_sparql = generate_sparql_with_qwen(qwen_tok, qwen_model, original_question, candidate_question, gold_sparql)
	if not pred_sparql:
		return False, {"ppl": ppl, "bertscore": bert_f1, "reason": "qwen_no_sparql"}

	perturbed_answers = run_sparql(sparql_endpoint, pred_sparql)
	if perturbed_answers is None:
		return False, {"ppl": ppl, "bertscore": bert_f1, "reason": "perturbed_sparql_failed"}

	ok = original_answers == perturbed_answers
	reason = "ok" if ok else "denotation_mismatch"
	return ok, {
		"ppl": ppl,
		"bertscore": bert_f1,
		"reason": reason,
		"predicted_sparql": pred_sparql,
		"gold_answer_size": len(original_answers),
		"pred_answer_size": len(perturbed_answers),
	}


def insert_webqsp_perturbed_field(record: Dict[str, Any], perturbed_question: str) -> OrderedDict:
	new_record: OrderedDict = OrderedDict()
	for key, value in record.items():
		new_record[key] = value
		if key == "ProcessedQuestion":
			new_record["Perturbed question"] = perturbed_question
	if "ProcessedQuestion" not in record:
		new_record["Perturbed question"] = perturbed_question
	return new_record


def mutate_record(record: Dict[str, Any], dataset_name: str, perturbed_question: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
	out = copy.deepcopy(record)
	if dataset_name == "cwq":
		out["question"] = perturbed_question
		out["Perturbed question"] = perturbed_question
	else:
		out = insert_webqsp_perturbed_field(out, perturbed_question)
	out["perturbation_meta"] = metrics
	return out


def process_records(
	records: List[Dict[str, Any]],
	dataset_name: str,
	prompt: PromptSpec,
	llama_tok: AutoTokenizer,
	llama_model: AutoModelForCausalLM,
	qwen_tok: AutoTokenizer,
	qwen_model: AutoModelForCausalLM,
	args: argparse.Namespace,
) -> List[Dict[str, Any]]:
	out = []
	limit = args.smoke_size if args.smoke_size and args.smoke_size > 0 else len(records)
	subset = records[:limit]
	for start in range(0, len(subset), args.batch_size):
		batch = subset[start : start + args.batch_size]
		questions = [extract_question(x, dataset_name) for x in batch]
		hints = [build_answer_hint(x, dataset_name) for x in batch]
		gold_sparqls = [extract_gold_sparql(x, dataset_name) for x in batch]
		attempts_left = [args.attempts] * len(batch)
		chosen: List[Optional[Tuple[str, Dict[str, Any]]]] = [None] * len(batch)

		while any(a > 0 and c is None for a, c in zip(attempts_left, chosen)):
			pending_idx = [i for i, (a, c) in enumerate(zip(attempts_left, chosen)) if a > 0 and c is None]
			messages = []
			for i in pending_idx:
				user_text = prompt.user.format(question=questions[i], answer_hint=hints[i])
				messages.append([
					{"role": "system", "content": prompt.system},
					{"role": "user", "content": user_text},
				])

			generations = batched_generate(llama_tok, llama_model, messages, max_new_tokens=args.max_new_tokens)
			for i, candidate in zip(pending_idx, generations):
				attempts_left[i] -= 1
				ok, metrics = evaluate_candidate(
					original_question=questions[i],
					candidate_question=candidate,
					gold_sparql=gold_sparqls[i],
					llama_tok=llama_tok,
					llama_model=llama_model,
					qwen_tok=qwen_tok,
					qwen_model=qwen_model,
					sparql_endpoint=args.sparql_endpoint,
					ppl_threshold=args.ppl_threshold,
					bert_threshold=args.bertscore_threshold,
				)
				if ok:
					chosen[i] = (candidate, metrics)

		for i, row in enumerate(batch):
			if chosen[i] is None:
				fallback = questions[i]
				metrics = {
					"reason": "constraint_unsatisfied_fallback_original",
					"ppl": compute_ppl(llama_tok, llama_model, fallback),
					"bertscore": 1.0,
				}
				out.append(mutate_record(row, dataset_name, fallback, metrics))
			else:
				perturbed, metrics = chosen[i]
				out.append(mutate_record(row, dataset_name, perturbed, metrics))
	return out


def read_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)


def is_cwq_file(path: Path) -> bool:
	return path.suffix == ".json" and "complexwebquestions" in path.name.lower()


def is_webqsp_file(path: Path) -> bool:
	return path.suffix == ".json" and "webqsp" in path.name.lower()


def iter_json_files(root: Path, matcher) -> List[Path]:
	return sorted([p for p in root.rglob("*.json") if matcher(p)])


def process_cwq_file(
	in_file: Path,
	out_file: Path,
	prompt: PromptSpec,
	llama_tok: AutoTokenizer,
	llama_model: AutoModelForCausalLM,
	qwen_tok: AutoTokenizer,
	qwen_model: AutoModelForCausalLM,
	args: argparse.Namespace,
) -> None:
	payload = read_json(in_file)
	if not isinstance(payload, list):
		raise ValueError(f"CWQ file must be a list: {in_file}")
	out_records = process_records(payload, "cwq", prompt, llama_tok, llama_model, qwen_tok, qwen_model, args)
	write_json(out_file, out_records)


def process_webqsp_file(
	in_file: Path,
	out_file: Path,
	prompt: PromptSpec,
	llama_tok: AutoTokenizer,
	llama_model: AutoModelForCausalLM,
	qwen_tok: AutoTokenizer,
	qwen_model: AutoModelForCausalLM,
	args: argparse.Namespace,
) -> None:
	payload = read_json(in_file)
	if isinstance(payload, dict) and "Questions" in payload:
		questions = payload["Questions"]
		out_questions = process_records(questions, "webqsp", prompt, llama_tok, llama_model, qwen_tok, qwen_model, args)
		new_payload = copy.deepcopy(payload)
		new_payload["Questions"] = out_questions
		write_json(out_file, new_payload)
		return

	if isinstance(payload, list):
		out_questions = process_records(payload, "webqsp", prompt, llama_tok, llama_model, qwen_tok, qwen_model, args)
		write_json(out_file, out_questions)
		return

	raise ValueError(f"Unexpected WebQSP schema: {in_file}")


def main() -> None:
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list

	print(f"Using GPUs: {args.gpu_list}")
	print(f"SPARQL endpoint: {args.sparql_endpoint}")

	llama_tok, llama_model = load_chat_model(args.llama_model)
	qwen_tok, qwen_model = load_chat_model(args.qwen_model)

	cwq_files = iter_json_files(args.cwq_dir, is_cwq_file)
	webqsp_files = iter_json_files(args.webqsp_dir, is_webqsp_file)

	if not cwq_files:
		raise RuntimeError(f"No CWQ json files found under: {args.cwq_dir}")
	if not webqsp_files:
		raise RuntimeError(f"No WebQSP json files found under: {args.webqsp_dir}")

	for prompt in PROMPTS:
		set_dir = args.out_root / f"set_{prompt.set_id}"
		for in_file in cwq_files:
			rel = in_file.relative_to(args.cwq_dir)
			out_file = set_dir / "cwq" / rel
			print(f"[set_{prompt.set_id}] CWQ: {in_file} -> {out_file}")
			process_cwq_file(in_file, out_file, prompt, llama_tok, llama_model, qwen_tok, qwen_model, args)

		for in_file in webqsp_files:
			rel = in_file.relative_to(args.webqsp_dir)
			out_file = set_dir / "webqsp" / rel
			print(f"[set_{prompt.set_id}] WebQSP: {in_file} -> {out_file}")
			process_webqsp_file(in_file, out_file, prompt, llama_tok, llama_model, qwen_tok, qwen_model, args)

	print("Completed perturbation set generation.")


if __name__ == "__main__":
	main()