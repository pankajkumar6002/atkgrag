#!/usr/bin/env python3
"""
Enhanced Seed Entity Extraction Agent with Freebase MID Mapping
Using Llama 3.1 8B via Transformers (instead of Ollama)
Based on "An Entity Linking Agent for Question Answering" (arXiv:2508.03865v3)

Supports:
- Seed entity extraction using Llama 3.1 8B
- Wikidata QID mapping
- Freebase MID mapping
- Offline operation with local database
- WebQSP (1639 test) and CWQ (3475 test) datasets
"""

import json
import requests
import sqlite3
import pickle
import os
import re
import torch
import transformers
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict


class EntityDatabase:
    """Local database for entity mappings (offline support)."""
    
    def __init__(self, db_path: str = "data/entity_mappings.db"):
        """Initialize database connection."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()
        
        # In-memory cache for faster lookups
        self.cache = {
            'name_to_qid': {},
            'qid_to_mid': {},
            'mid_to_qid': {}
        }
        
    def _create_tables(self):
        """Create database tables if they don't exist."""
        # Freebase MID <-> Wikidata QID mapping
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS freebase_wikidata (
                freebase_mid TEXT PRIMARY KEY,
                wikidata_qid TEXT NOT NULL
            )
        ''')
        
        # Entity name <-> Wikidata QID mapping (for offline use)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_names (
                entity_name TEXT,
                wikidata_qid TEXT NOT NULL,
                aliases TEXT,
                description TEXT,
                score REAL DEFAULT 1.0,
                PRIMARY KEY (entity_name, wikidata_qid)
            )
        ''')
        
        # Create indexes for faster lookups
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_qid 
            ON freebase_wikidata(wikidata_qid)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_entity_name 
            ON entity_names(entity_name)
        ''')
        
        self.conn.commit()
    
    def load_freebase_wikidata_mapping(self, mapping_data):
        """
        Load Freebase MID to Wikidata QID mappings from HuggingFace dataset.
        
        Args:
            mapping_data: Dataset from HuggingFace or dict/list of mappings
        """
        print(f"Loading Freebase/Wikidata mappings...")
        count = 0
        batch = []
        
        # Handle HuggingFace dataset format
        if hasattr(mapping_data, '__iter__'):
            for item in mapping_data:
                freebase_id = item.get('freebase_id', '')
                wikidata_id = item.get('wikidata_id', '')
                
                if freebase_id and wikidata_id:
                    # Ensure proper format
                    if not freebase_id.startswith('/m/'):
                        freebase_id = f"/m/{freebase_id.replace('m.', '')}"
                    if not wikidata_id.startswith('Q'):
                        wikidata_id = f"Q{wikidata_id}"
                    
                    batch.append((freebase_id, wikidata_id))
                    count += 1
                    
                    # Batch insert for performance
                    if len(batch) >= 10000:
                        self.cursor.executemany(
                            'INSERT OR REPLACE INTO freebase_wikidata VALUES (?, ?)',
                            batch
                        )
                        self.conn.commit()
                        batch = []
                        print(f"  Loaded {count:,} mappings...")
        
        # Insert remaining
        if batch:
            self.cursor.executemany(
                'INSERT OR REPLACE INTO freebase_wikidata VALUES (?, ?)',
                batch
            )
            self.conn.commit()
        
        print(f"✓ Loaded {count:,} Freebase/Wikidata mappings")
    
    def get_qid_from_mid(self, freebase_mid: str) -> Optional[str]:
        """Get Wikidata QID from Freebase MID."""
        # Normalize format
        if not freebase_mid.startswith('/m/'):
            freebase_mid = f"/m/{freebase_mid.replace('m.', '')}"
        
        # Check cache first
        if freebase_mid in self.cache['mid_to_qid']:
            return self.cache['mid_to_qid'][freebase_mid]
        
        # Query database
        self.cursor.execute(
            'SELECT wikidata_qid FROM freebase_wikidata WHERE freebase_mid = ?',
            (freebase_mid,)
        )
        result = self.cursor.fetchone()
        
        if result:
            qid = result[0]
            self.cache['mid_to_qid'][freebase_mid] = qid
            return qid
        
        return None
    
    def get_mid_from_qid(self, wikidata_qid: str) -> Optional[str]:
        """Get Freebase MID from Wikidata QID."""
        # Check cache first
        if wikidata_qid in self.cache['qid_to_mid']:
            return self.cache['qid_to_mid'][wikidata_qid]
        
        # Query database
        self.cursor.execute(
            'SELECT freebase_mid FROM freebase_wikidata WHERE wikidata_qid = ?',
            (wikidata_qid,)
        )
        result = self.cursor.fetchone()
        
        if result:
            mid = result[0]
            self.cache['qid_to_mid'][wikidata_qid] = mid
            return mid
        
        return None
    
    def search_entity(self, entity_name: str, limit: int = 5) -> List[Dict]:
        """Search for entity by name using local entity_names table (offline).

        The entity_names table is keyed by name_lower (lowercased Freebase label).
        Returns list of dicts with 'mid' (m.XXXX format) and 'label'.
        Falls back to prefix search if exact match returns nothing.
        """
        entity_name_normalized = entity_name.lower().strip()

        cache_key = f"{entity_name_normalized}:{limit}"
        if cache_key in self.cache['name_to_qid']:
            return self.cache['name_to_qid'][cache_key]

        # Exact match on lowercased name
        self.cursor.execute(
            "SELECT mid, entity_name FROM entity_names WHERE name_lower = ? LIMIT ?",
            (entity_name_normalized, limit),
        )
        rows = self.cursor.fetchall()

        # Prefix fallback
        if not rows:
            self.cursor.execute(
                "SELECT mid, entity_name FROM entity_names WHERE name_lower LIKE ? LIMIT ?",
                (entity_name_normalized[:20] + "%", limit),
            )
            rows = self.cursor.fetchall()

        results = []
        for mid, label in rows:
            # Also resolve QID for compatibility
            qid = self.get_qid_from_mid(mid) or ""
            results.append({
                'mid': mid,       # m.XXXX format — primary key for GNN-RAG
                'qid': qid,
                'label': label,
                'description': '',
                'aliases': [],
                'score': 1.0,
            })

        self.cache['name_to_qid'][cache_key] = results
        return results
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class WikidataClient:
    """Client for Wikidata API (online mode)."""
    
    def __init__(self):
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.session = requests.Session()
    
    def search_entity(self, entity_name: str, limit: int = 5) -> List[Dict]:
        """Search for entity on Wikidata."""
        params = {
            'action': 'wbsearchentities',
            'search': entity_name,
            'language': 'en',
            'limit': limit,
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('search', []):
                results.append({
                    'qid': item['id'],
                    'label': item.get('label', ''),
                    'aliases': item.get('aliases', []),
                    'description': item.get('description', ''),
                    'score': 1.0
                })
            
            return results
            
        except Exception as e:
            print(f"Warning: Wikidata API error: {e}")
            return []
    
    def get_freebase_mid(self, wikidata_qid: str) -> Optional[str]:
        """Get Freebase MID from Wikidata entity (via P646 property)."""
        params = {
            'action': 'wbgetclaims',
            'entity': wikidata_qid,
            'property': 'P646',  # Freebase ID property
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            claims = data.get('claims', {}).get('P646', [])
            if claims:
                mid = claims[0]['mainsnak']['datavalue']['value']
                if not mid.startswith('/m/'):
                    mid = f"/m/{mid}"
                return mid
            
        except Exception as e:
            print(f"Warning: Failed to get Freebase MID for {wikidata_qid}: {e}")
        
        return None


class SeedEntityAgentWithMapping:
    """
    Enhanced seed entity extraction agent with Freebase MID mapping.
    Uses Llama 3.1 8B via Transformers.

    Key fixes vs. original:
    - Extracts ALL entities per question (not just primary seed).
      WebQSP questions often have 2 topic entities; both are needed for
      a complete subgraph.  All entities are linked and returned in
      `linked_entities: [{mention, mid, source, ...}]`.
    - Three-path MID resolution (offline-first, widest coverage):
        Path 1: title2freebase_mid.json  (22.9M Wikipedia-title → MID)
        Path 2: entities_sup_names name2mid  (11.3M Freebase-label → MID)
        Path 3: qid_to_mid.pkl  (2.1M Wikidata-QID → MID, via fb2w.nt.gz inverse)
        Path 4: Wikidata API  (online fallback; only when offline_mode=False)
    - MID normalised to m.XXXX format (GNN-RAG convention).
    - If a NER model is provided (ner_model arg), it is called first for span
      detection; LLM is used only for disambiguation of NER candidates.
    """

    # Offline resource paths
    _TITLE2MID_PATH  = REPO_ROOT / "external" / "BLINK" / "models" / "title2freebase_mid.json"
    _QID2MID_PATH    = REPO_ROOT / "data" / "mid_cache" / "qid_to_mid.pkl"
    _SUP_NAMES_PATH  = REPO_ROOT / "data" / "resources" / "entities_sup_names.tsv"

    def __init__(
        self,
        model_id: str = str(REPO_ROOT / "models" / "Llama-3.1-8B-Instruct"),
        db_path: str = str(REPO_ROOT / "data" / "resources" / "entity_mappings.db"),
        offline_mode: bool = True,
        device: str = "auto",
        verbose: bool = False,
        ner_model=None,          # Optional: pre-loaded NER model for span detection
    ):
        self.offline_mode = offline_mode
        self.verbose = verbose
        self.ner_model = ner_model  # if provided, used for Step 1 span detection

        # ── LLM pipeline (Llama-3.1-8B) ──────────────────────────────────────
        print(f"Loading {model_id} …")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device,
        )
        print("✓ Model loaded")

        # ── Local entity DB (SQLite) ──────────────────────────────────────────
        self.db = EntityDatabase(db_path)

        # ── Path 1: title2freebase_mid.json (22.9M Wikipedia titles) ─────────
        self.title2mid: Dict[str, str] = {}
        if self._TITLE2MID_PATH.exists():
            print(f"Loading title2freebase_mid.json ({self._TITLE2MID_PATH.stat().st_size >> 20} MB) …")
            with open(self._TITLE2MID_PATH) as fh:
                self.title2mid = json.load(fh)
            print(f"  {len(self.title2mid):,} entries")
        else:
            print(f"[WARN] title2freebase_mid.json not found — Path 1 disabled")

        # ── Path 3: qid_to_mid.pkl (2.1M QID → MID from fb2w.nt.gz inverse) ─
        self.qid2mid: Dict[str, str] = {}
        if self._QID2MID_PATH.exists():
            import pickle
            with open(self._QID2MID_PATH, "rb") as fh:
                self.qid2mid = pickle.load(fh)
            print(f"Loaded qid_to_mid: {len(self.qid2mid):,} entries (fb2w inverse)")
        else:
            print(f"[WARN] qid_to_mid.pkl not found — Path 3 disabled")

        # ── Path 2: entities_sup_names name2mid (11.3M Freebase labels) ──────
        # Loaded lazily from DB if SQLite entity_names table is populated;
        # otherwise fall back to TSV scan on first miss.
        self._name2mid_cache: Dict[str, str] = {}
        self._name2mid_loaded = False

        # ── Wikidata client (Path 4 — online fallback) ────────────────────────
        self.wikidata_client = WikidataClient() if not offline_mode else None

        # ── Result cache ──────────────────────────────────────────────────────
        self.results_cache: Dict[str, Dict] = {}
    
    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[Agent] {message}")
    
    def _call_llm(self, prompt: str) -> str:
        """Call Llama model via Transformers pipeline."""
        try:
            # Format prompt for Llama 3.1 Instruct
            messages = [
                {"role": "system", "content": "You are an expert entity extraction agent."},
                {"role": "user", "content": prompt}
            ]
            
            # Generate response
            outputs = self.pipeline(
                messages,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
            )
            
            # Extract generated text
            response = outputs[0]["generated_text"][-1]["content"]
            return response.strip()
            
        except Exception as e:
            raise RuntimeError(f"Error calling Llama model: {e}")
    
    def _build_extraction_prompt(self, query: str, ner_spans: List[str] = None) -> str:
        """Build CoT prompt to extract ALL entities needed to answer the question.

        For KBQA the subgraph is built from the union of neighborhoods of ALL
        topic entities (not just the primary one).  Two-hop WebQSP questions
        routinely need 2 seed entities (e.g. "who plays X in show Y" → both
        character X and show Y are seeds).

        If ner_spans is provided (from an external NER model), they are listed
        as pre-identified candidates and the LLM only needs to disambiguate.
        """
        ner_block = ""
        if ner_spans:
            ner_block = (
                f"\nPre-identified entity spans (from NER): {ner_spans}\n"
                "Confirm or correct these, and add any missed entities.\n"
            )

        prompt = f"""You are an expert entity extraction agent for knowledge-base question answering (KBQA).
Your task: identify ALL named entities in the query that are SEED ENTITIES — i.e., entities
whose Freebase neighborhood must be retrieved to answer the question.
{ner_block}
IMPORTANT: Many KBQA questions have TWO seed entities (e.g. "who plays [character] in [show]"
needs both the character and the show as seeds). Extract ALL of them.

Query: "{query}"

Think step-by-step:
1. Identify every named entity mention (people, places, works, organizations, events).
2. Decide which are SEED entities (needed for subgraph retrieval) vs. answer slots.
3. For each seed entity, write its canonical/Wikipedia-style name.

Respond in JSON:
{{
  "reasoning": "brief step-by-step reasoning",
  "seed_entities": [
    {{"mention": "exact span in query", "canonical": "Wikipedia/canonical name", "type": "person|place|work|org|event|other"}},
    ...
  ],
  "confidence": "high|medium|low"
}}

Return ONLY valid JSON, nothing else."""
        return prompt

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response. Returns seed_entities list + metadata."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                seed_entities = parsed.get("seed_entities", [])
                # Normalise to list of dicts with mention + canonical
                normalised = []
                for ent in seed_entities:
                    if isinstance(ent, str):
                        normalised.append({"mention": ent, "canonical": ent, "type": "unknown"})
                    elif isinstance(ent, dict):
                        normalised.append({
                            "mention":   ent.get("mention", ent.get("entity", "")),
                            "canonical": ent.get("canonical", ent.get("mention", "")),
                            "type":      ent.get("type", "unknown"),
                        })
                # Back-compat: also parse legacy "all_entities" / "seed_entity" fields
                if not normalised:
                    for e in parsed.get("all_entities", []):
                        normalised.append({"mention": e, "canonical": e, "type": "unknown"})
                if not normalised and parsed.get("seed_entity"):
                    e = parsed["seed_entity"]
                    normalised.append({"mention": e, "canonical": e, "type": "unknown"})
                return {
                    "seed_entities": normalised,
                    "reasoning":     parsed.get("reasoning", ""),
                    "confidence":    parsed.get("confidence", "unknown"),
                }
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: extract quoted strings as entity mentions
        mentions = re.findall(r'"([^"]{2,60})"', response)
        return {
            "seed_entities": [{"mention": m, "canonical": m, "type": "unknown"} for m in mentions[:5]],
            "reasoning": response[:200],
            "confidence": "low",
        }
    
    def _normalize_mid(self, mid: Optional[str]) -> Optional[str]:
        """Normalise any MID format to m.XXXX (GNN-RAG convention)."""
        if not mid:
            return None
        return mid.lstrip("/").replace("/", ".")

    def _lookup_name2mid(self, name: str) -> Optional[str]:
        """Path-2 lookup: Freebase label → MID via entities_sup_names.tsv.

        Loads entire TSV into memory on first call (11.3M entries, ~1 GB RAM).
        Uses DB entity_names table if already populated; falls back to TSV scan.
        """
        key = name.lower().strip()
        if key in self._name2mid_cache:
            return self._name2mid_cache[key]

        if not self._name2mid_loaded:
            # Try SQLite first (fast if entity_names table was populated by build_entity_db.py)
            self.db.cursor.execute(
                "SELECT mid FROM entity_names WHERE name_lower = ? LIMIT 1", (key,)
            )
            row = self.db.cursor.fetchone()
            if row:
                return row[0]

            # Fall back to TSV scan if DB table empty
            if self._SUP_NAMES_PATH.exists():
                print("  [name2mid] Loading entities_sup_names.tsv into cache …", flush=True)
                with open(self._SUP_NAMES_PATH, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        parts = line.rstrip("\n").split("\t", 1)
                        if len(parts) == 2:
                            mid, nm = parts
                            k = nm.lower().strip()
                            if k and k not in self._name2mid_cache:
                                self._name2mid_cache[k] = mid
                self._name2mid_loaded = True
                return self._name2mid_cache.get(key)
        return self._name2mid_cache.get(key)

    def map_entity_to_ids(self, entity_name: str, mention: str = "") -> Dict:
        """Map entity name → Freebase MID using local DB (offline-first).

        Three-path offline resolution + one online fallback.
        Both `entity_name` (canonical/Wikipedia form) and `mention` (raw span)
        are tried at each path.  MID returned in m.XXXX format.

        Path 1 — title2freebase_mid.json (22.9M Wikipedia titles, widest)
        Path 2 — entities_sup_names name2mid (11.3M Freebase labels)
        Path 3 — qid_to_mid.pkl via fb2w.nt.gz inverse (2.1M, most authoritative)
                  Requires a Wikidata QID first; obtained from SQLite or API.
        Path 4 — Wikidata API (online only; returns QID → Path 3)
        """
        self._log(f"Mapping: canonical={entity_name!r}  mention={mention!r}")

        def _try_names(*names):
            """Try each name variant through Paths 1 and 2."""
            for nm in names:
                if not nm:
                    continue
                # Path 1: Wikipedia title → MID
                mid = self._normalize_mid(
                    self.title2mid.get(nm.lower()) or self.title2mid.get(nm)
                )
                if mid:
                    return mid, "title2freebase_mid"
                # Path 2: Freebase label → MID
                mid = self._normalize_mid(self._lookup_name2mid(nm))
                if mid:
                    return mid, "entities_sup_names"
            return None, None

        mid, source = _try_names(entity_name, mention)
        if mid:
            self._log(f"  Resolved via {source}: {mid}")
            return {'qid': None, 'mid': mid, 'label': entity_name,
                    'description': '', 'aliases': [], 'source': source, 'candidates': []}

        # Path 3: try SQLite entity_names → QID → qid_to_mid.pkl
        db_results = self.db.search_entity(entity_name, limit=3)
        for res in db_results:
            qid = res.get('qid', '')
            if qid:
                mid = self._normalize_mid(self.qid2mid.get(qid))
                if mid:
                    self._log(f"  Resolved via qid_to_mid (SQLite→QID→PKL): {mid}")
                    return {'qid': qid, 'mid': mid, 'label': res.get('label', entity_name),
                            'description': '', 'aliases': [], 'source': 'qid_to_mid_pkl',
                            'candidates': db_results}

        if self.offline_mode:
            self._log("  All offline paths failed.")
            return {'qid': None, 'mid': None, 'label': entity_name,
                    'description': '', 'aliases': [], 'source': 'none', 'candidates': []}

        # Path 4: Wikidata API → QID → qid_to_mid.pkl (online fallback)
        self._log("  Querying Wikidata API …")
        wikidata_results = self.wikidata_client.search_entity(entity_name, limit=5)
        for res in wikidata_results:
            qid = res['qid']
            mid = self._normalize_mid(
                self.qid2mid.get(qid) or self.db.get_mid_from_qid(qid)
                or self._normalize_mid(self.wikidata_client.get_freebase_mid(qid))
            )
            if mid:
                self._log(f"  Resolved via Wikidata API → qid_to_mid: {mid}")
                return {'qid': qid, 'mid': mid,
                        'label': res.get('label', entity_name),
                        'description': res.get('description', ''),
                        'aliases': res.get('aliases', []),
                        'source': 'wikidata_api', 'candidates': wikidata_results}

        return {'qid': None, 'mid': None, 'label': entity_name,
                'description': '', 'aliases': [], 'source': 'not_found', 'candidates': []}
    
    def extract_and_link(self, query: str, gold_mids: List[str] = None) -> Dict:
        """Extract ALL seed entities and link each to a Freebase MID.

        Args:
            query:     Input question.
            gold_mids: List of gold Freebase MIDs (for evaluation).
                       Accepts both m.XXXX and /m/XXXX formats.

        Returns:
            {
              query, reasoning, confidence,
              linked_entities: [
                {mention, canonical, type, mid, qid, source, verified}, ...
              ],
              eval: {gold_mids, any_correct, all_correct, found_count}  # if gold_mids provided
            }
        """
        self._log(f"Processing: {query}")

        if query in self.results_cache:
            return self.results_cache[query]

        # ── Step 1: Entity span detection ────────────────────────────────────
        ner_spans: List[str] = []
        if self.ner_model is not None:
            try:
                ner_spans = self.ner_model(query)  # expected: list of span strings
                self._log(f"NER spans: {ner_spans}")
            except Exception as e:
                self._log(f"NER failed: {e}")

        prompt   = self._build_extraction_prompt(query, ner_spans or None)
        response = self._call_llm(prompt)
        parsed   = self._parse_response(response)

        seed_entities = parsed["seed_entities"]
        self._log(f"Extracted {len(seed_entities)} seed entities: {[e['mention'] for e in seed_entities]}")

        # ── Step 2: Link each entity through 3-path resolution ───────────────
        linked: List[Dict] = []
        for ent in seed_entities:
            mention   = ent.get("mention", "")
            canonical = ent.get("canonical", mention)
            mapping   = self.map_entity_to_ids(canonical, mention=mention)
            mid       = mapping["mid"]
            # Local verification: mid ∈ entities_sup is checked inside map_entity_to_ids;
            # here we just flag whether a MID was resolved at all.
            linked.append({
                "mention":   mention,
                "canonical": canonical,
                "type":      ent.get("type", "unknown"),
                "mid":       mid,
                "qid":       mapping.get("qid"),
                "source":    mapping["source"],
                "label":     mapping.get("label", canonical),
            })

        # ── Step 3: Evaluation against gold MIDs (if provided) ───────────────
        eval_block: Dict = {}
        if gold_mids:
            gold_norm = {self._normalize_mid(g) or g for g in gold_mids}
            found_mids = {e["mid"] for e in linked if e["mid"]}
            any_correct = bool(found_mids & gold_norm)
            all_correct = gold_norm.issubset(found_mids)
            eval_block = {
                "gold_mids":   list(gold_norm),
                "found_mids":  list(found_mids),
                "any_correct": any_correct,
                "all_correct": all_correct,
                "found_count": len(found_mids),
            }
            self._log(f"Eval: any_correct={any_correct} all_correct={all_correct}")

        result = {
            "query":           query,
            "reasoning":       parsed["reasoning"],
            "confidence":      parsed["confidence"],
            "linked_entities": linked,
            **({"eval": eval_block} if eval_block else {}),
        }
        self.results_cache[query] = result
        return result
    
    def process_dataset(
        self,
        dataset_file: str,
        output_file: str,
        dataset_type: str = "webqsp"
    ) -> Dict:
        """
        Process entire dataset (WebQSP or CWQ).
        
        Args:
            dataset_file: Path to dataset JSON file
            output_file: Path to save results
            dataset_type: "webqsp" or "cwq"
            
        Returns:
            Summary statistics
        """
        print(f"\n{'='*70}")
        print(f"Processing {dataset_type.upper()} Dataset")
        print(f"{'='*70}\n")
        
        # Load dataset
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract questions based on dataset format
        if dataset_type.lower() == "webqsp":
            questions = data.get('Questions', [])
        else:  # cwq - array format
            questions = data if isinstance(data, list) else [data]
        
        total = len(questions)
        print(f"Loaded {total} questions\n")
        
        results = []
        stats = {
            'total': total,
            'processed': 0,
            'linking_correct': 0,
            'linking_found': 0,
            'no_gold': 0
        }
        
        for i, question_data in enumerate(questions, 1):
            # ── Extract question text and ALL gold MIDs ───────────────────────
            if dataset_type.lower() == "webqsp":
                query = question_data.get('ProcessedQuestion',
                                          question_data.get('RawQuestion', ''))
                # Collect ALL TopicEntityMids across all Parses (dedup)
                gold_mids = list({
                    p['TopicEntityMid']
                    for p in question_data.get('Parses', [])
                    if p.get('TopicEntityMid')
                })
            else:  # cwq — perturbed dataset canonical format
                query     = question_data.get('perturbed_question',
                                              question_data.get('question', ''))
                raw_sparql = question_data.get('sparql', '')
                import re as _re
                gold_mids = list(set(_re.findall(r'ns:(m\.[^\s>)]+)', raw_sparql)))
                if not gold_mids and question_data.get('topic_entity_mid'):
                    gold_mids = [question_data['topic_entity_mid']]

            print(f"[{i}/{total}] {query[:80]}  (gold_mids={gold_mids})")

            try:
                result = self.extract_and_link(query, gold_mids=gold_mids or None)
                results.append(result)

                stats['processed'] += 1
                found = [e for e in result['linked_entities'] if e['mid']]
                if found:
                    stats['linking_found'] += 1
                if gold_mids:
                    ev = result.get('eval', {})
                    if ev.get('any_correct'):
                        stats['linking_correct'] += 1
                else:
                    stats['no_gold'] += 1

                if i % 10 == 0 or i == total:
                    denom = stats['processed'] - stats['no_gold']
                    accuracy = (stats['linking_correct'] / denom * 100) if denom > 0 else 0
                    coverage = stats['linking_found'] / stats['processed'] * 100
                    print(f"  [{i}/{total}] any_correct={accuracy:.1f}%  coverage={coverage:.1f}%")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({'query': query, 'error': str(e), 'gold_mids': gold_mids})
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset_type,
                'statistics': stats,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*70}")
        print("Summary Statistics")
        print(f"{'='*70}")
        print(f"Total Questions:        {stats['total']}")
        print(f"Successfully Processed: {stats['processed']}")
        print(f"Entity Linking Found:   {stats['linking_found']} ({stats['linking_found']/stats['total']*100:.1f}%)")
        
        if stats['processed'] - stats['no_gold'] > 0:
            accuracy = stats['linking_correct'] / (stats['processed'] - stats['no_gold']) * 100
            print(f"Entity Linking Accuracy: {stats['linking_correct']}/{stats['processed'] - stats['no_gold']} ({accuracy:.1f}%)")
        
        print(f"{'='*70}\n")
        
        return stats
    
    def close(self):
        """Clean up resources."""
        self.db.close()


def main():
    """Example usage."""
    print("="*70)
    print("Seed Entity Extraction Agent with Freebase MID Mapping")
    print("Using Llama 3.1 8B via Transformers")
    print("="*70)
    
    # Initialize agent
    agent = SeedEntityAgentWithMapping(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        offline_mode=False,
        verbose=True
    )
    
    # Single query example
    print("\n" + "="*70)
    print("Single Query Example")
    print("="*70 + "\n")
    
    query = "What movies did the actor who played Iron Man star in?"
    result = agent.extract_and_link(query)
    
    print(f"Query: {result['query']}")
    print(f"Seed Entity: {result['seed_entity']}")
    print(f"Wikidata QID: {result['wikidata_qid']}")
    print(f"Freebase MID: {result['freebase_mid']}")
    print(f"Description: {result['entity_description']}")
    print(f"Confidence: {result['confidence']}")
    
    # Clean up
    agent.close()


if __name__ == "__main__":
    main()
