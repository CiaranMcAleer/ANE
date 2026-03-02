#!/usr/bin/env python3
"""
generate.py — Synthetic ER extraction training data via local Ollama LLM.

Usage:
    python3 generate.py                         # use default model + settings
    python3 generate.py --model qwen2.5:7b      # specify model
    python3 generate.py --per-domain 100        # examples per domain
    python3 generate.py --output data.jsonl     # output path
    python3 generate.py --resume                # skip already-generated domains
"""

import json
import random
import time
import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
OUTPUT_FILE  = "er_training_data.jsonl"
N_PER_DOMAIN = 50
MAX_RETRIES  = 3

# ── Domain Definitions ──────────────────────────────────────────────────────
# Each domain has: a schema (entity + relation types) and stylistic hints
# so the generated text is realistic and varied.

DOMAINS = [
    {
        "name": "organizational_hr",
        "description": "Corporate org structures, employees, departments, projects",
        "entity_types": ["Person", "Department", "Role", "Organization", "Project", "Skill", "Location"],
        "relation_types": ["WORKS_IN", "REPORTS_TO", "HAS_ROLE", "LEADS", "MEMBER_OF",
                           "ASSIGNED_TO", "HAS_SKILL", "LOCATED_IN", "COLLABORATES_WITH"],
        "style_hints": "HR notes, org chart descriptions, team briefings, performance reviews, onboarding docs"
    },
    {
        "name": "healthcare",
        "description": "Patients, clinicians, conditions, treatments, facilities",
        "entity_types": ["Patient", "Doctor", "Nurse", "Hospital", "Condition",
                         "Medication", "Procedure", "Department", "Specialist"],
        "relation_types": ["TREATED_BY", "HAS_CONDITION", "PRESCRIBED", "ADMITTED_TO",
                           "PERFORMED", "WORKS_AT", "REFERRED_TO", "ALLERGIC_TO"],
        "style_hints": "clinical notes, discharge summaries, referral letters, care plans"
    },
    {
        "name": "legal_contracts",
        "description": "Legal entities, contracts, obligations, assets, jurisdictions",
        "entity_types": ["Person", "Company", "Contract", "Asset", "Obligation",
                         "Court", "Jurisdiction", "Clause", "Attorney"],
        "relation_types": ["PARTY_TO", "OWNS", "OBLIGATED_BY", "GOVERNS",
                           "FILED_IN", "REPRESENTS", "TRANSFERS", "SUPERSEDES"],
        "style_hints": "contract recitals, legal briefs, settlement agreements, terms of service"
    },
    {
        "name": "finance_business",
        "description": "Companies, transactions, assets, markets, financial instruments",
        "entity_types": ["Company", "Person", "Asset", "Transaction", "Fund",
                         "Market", "Currency", "Bank", "Index"],
        "relation_types": ["OWNS", "INVESTS_IN", "ACQUIRED_BY", "TRADED_ON",
                           "ISSUED_BY", "MANAGES", "SUBSIDIARY_OF", "PARTNERED_WITH"],
        "style_hints": "financial filings, earnings reports, merger announcements, analyst notes"
    },
    {
        "name": "academic_research",
        "description": "Researchers, papers, institutions, grants, findings",
        "entity_types": ["Researcher", "Paper", "Institution", "Grant",
                         "Dataset", "Method", "Field", "Conference", "Journal"],
        "relation_types": ["AUTHORED_BY", "AFFILIATED_WITH", "CITES", "FUNDED_BY",
                           "PUBLISHED_IN", "USES_METHOD", "PART_OF_FIELD", "REPLICATES"],
        "style_hints": "paper abstracts, grant proposals, lab descriptions, research summaries"
    },
    {
        "name": "software_technology",
        "description": "Software systems, components, developers, organizations, dependencies",
        "entity_types": ["System", "Component", "Developer", "Organization",
                         "Library", "API", "Protocol", "Language", "Platform"],
        "relation_types": ["DEPENDS_ON", "DEVELOPED_BY", "INTEGRATES_WITH",
                           "OWNED_BY", "USES", "EXPOSES", "SUPERSEDES", "FORKS_FROM"],
        "style_hints": "technical documentation, architecture docs, changelogs, engineering blogs"
    },
    {
        "name": "supply_chain_logistics",
        "description": "Suppliers, products, warehouses, shipments, routes",
        "entity_types": ["Supplier", "Product", "Warehouse", "Carrier", "Order",
                         "Customer", "Port", "Route", "Certificate"],
        "relation_types": ["SUPPLIES", "SHIPS_TO", "STORED_AT", "CARRIED_BY",
                           "ORDERED_BY", "ROUTED_THROUGH", "CERTIFIED_FOR", "SOURCED_FROM"],
        "style_hints": "shipping manifests, procurement docs, vendor agreements, logistics reports"
    },
    {
        "name": "real_estate_property",
        "description": "Properties, owners, agents, transactions, locations",
        "entity_types": ["Property", "Person", "Company", "Agent", "Transaction",
                         "Location", "Lease", "Lender", "Zone"],
        "relation_types": ["OWNS", "LISTED_BY", "SOLD_TO", "LOCATED_IN",
                           "LEASED_BY", "FINANCED_BY", "ADJACENT_TO", "ZONED_AS"],
        "style_hints": "property listings, title documents, lease agreements, development plans"
    },
    {
        "name": "government_policy",
        "description": "Government bodies, officials, policies, legislation, programs",
        "entity_types": ["Official", "Agency", "Policy", "Legislation",
                         "Program", "Committee", "Region", "Budget", "Treaty"],
        "relation_types": ["OVERSEES", "ENACTED_BY", "FUNDS", "APPLIES_TO",
                           "CHAIRS", "GOVERNS", "AMENDS", "RATIFIED_BY"],
        "style_hints": "policy briefs, legislative summaries, government reports, press releases"
    },
    {
        "name": "education",
        "description": "Schools, students, teachers, courses, qualifications",
        "entity_types": ["Student", "Teacher", "Institution", "Course",
                         "Qualification", "Department", "Program", "Assessment"],
        "relation_types": ["ENROLLED_IN", "TEACHES", "OFFERED_BY", "LEADS_TO",
                           "ASSESSED_BY", "PART_OF", "PREREQUISITE_FOR", "ACCREDITED_BY"],
        "style_hints": "course descriptions, academic transcripts, accreditation reports, curriculum docs"
    },
    {
        "name": "news_media",
        "description": "People, events, organizations in news contexts",
        "entity_types": ["Person", "Organization", "Event", "Location",
                         "Publication", "Role", "Date", "Topic"],
        "relation_types": ["INVOLVED_IN", "REPORTED_BY", "HELD_AT", "AFFILIATED_WITH",
                           "OCCURRED_IN", "CAUSED_BY", "RESPONDED_TO", "ANNOUNCED"],
        "style_hints": "news articles, press releases, editorial summaries, wire reports"
    },
    {
        "name": "science_environment",
        "description": "Species, ecosystems, environmental phenomena, researchers, policies",
        "entity_types": ["Species", "Ecosystem", "Phenomenon", "Researcher",
                         "Agency", "Compound", "Location", "Program"],
        "relation_types": ["INHABITS", "AFFECTS", "STUDIED_BY", "REGULATES",
                           "CAUSED_BY", "FOUND_IN", "THREATENS", "MONITORS"],
        "style_hints": "environmental impact reports, ecology papers, conservation plans, field notes"
    },
    {
        "name": "manufacturing_engineering",
        "description": "Factories, processes, materials, machinery, engineers",
        "entity_types": ["Facility", "Process", "Material", "Machine",
                         "Engineer", "Product", "Standard", "Vendor"],
        "relation_types": ["PRODUCES", "REQUIRES", "OPERATED_BY", "CERTIFIED_TO",
                           "SOURCED_FROM", "QUALITY_CHECKED_BY", "UPGRADED_FROM", "USED_IN"],
        "style_hints": "manufacturing specs, process documentation, quality audits, plant reports"
    },
    {
        "name": "transportation_infrastructure",
        "description": "Transport networks, vehicles, operators, routes, regulations",
        "entity_types": ["Vehicle", "Operator", "Route", "Station",
                         "Regulator", "Network", "Service", "Region"],
        "relation_types": ["OPERATES_ON", "MANAGED_BY", "CONNECTS", "REGULATED_BY",
                           "SERVES", "PART_OF_NETWORK", "LICENSED_BY", "DEPARTS_FROM"],
        "style_hints": "transit authority docs, infrastructure plans, timetable specs, safety reports"
    },
    {
        "name": "sports_entertainment",
        "description": "Athletes, teams, competitions, venues, sponsors",
        "entity_types": ["Athlete", "Team", "Competition", "Venue",
                         "Sponsor", "Coach", "Federation", "Award"],
        "relation_types": ["PLAYS_FOR", "COACHED_BY", "COMPETES_IN", "HELD_AT",
                           "SPONSORED_BY", "GOVERNED_BY", "WON", "TRANSFERRED_TO"],
        "style_hints": "sports reports, transfer announcements, federation rules, sponsorship docs"
    },
    {
        "name": "food_agriculture",
        "description": "Farms, crops, producers, distributors, certifications",
        "entity_types": ["Farm", "Crop", "Producer", "Distributor",
                         "Retailer", "Certification", "Region", "Season"],
        "relation_types": ["GROWS", "SUPPLIED_BY", "CERTIFIED_AS", "SOLD_TO",
                           "PRODUCED_IN", "HARVESTED_IN", "DISTRIBUTED_BY", "CONTRACTED_WITH"],
        "style_hints": "farm records, supply chain docs, certification reports, agricultural summaries"
    },
    {
        "name": "cybersecurity",
        "description": "Systems, threats, actors, vulnerabilities, mitigations",
        "entity_types": ["System", "ThreatActor", "Vulnerability", "Malware",
                         "Organization", "CVE", "Protocol", "Control"],
        "relation_types": ["EXPLOITS", "TARGETS", "MITIGATED_BY", "AFFECTS",
                           "ATTRIBUTED_TO", "PATCHES", "COMMUNICATES_VIA", "DISCOVERED_BY"],
        "style_hints": "threat intelligence reports, security advisories, incident reports, CVE descriptions"
    },
    {
        "name": "social_nonprofit",
        "description": "Nonprofits, communities, programs, beneficiaries, donors",
        "entity_types": ["Organization", "Person", "Program", "Community",
                         "Donor", "Beneficiary", "Grant", "Partner"],
        "relation_types": ["RUNS", "SERVES", "FUNDED_BY", "PARTNERS_WITH",
                           "BENEFITS_FROM", "VOLUNTEERS_AT", "DELIVERED_IN", "SUPPORTS"],
        "style_hints": "grant reports, charity filings, program evaluations, impact summaries"
    },
    {
        "name": "pharmaceutical_biotech",
        "description": "Drugs, trials, companies, researchers, regulators",
        "entity_types": ["Drug", "Trial", "Company", "Researcher",
                         "Regulator", "Compound", "Disease", "Biomarker"],
        "relation_types": ["DEVELOPED_BY", "TREATS", "APPROVED_BY", "TARGETS",
                           "CONDUCTED_BY", "SPONSORED_BY", "INHIBITS", "INDICATED_FOR"],
        "style_hints": "clinical trial summaries, FDA filings, drug labels, research abstracts"
    },
    {
        "name": "architecture_construction",
        "description": "Buildings, contractors, architects, materials, regulations",
        "entity_types": ["Building", "Contractor", "Architect", "Material",
                         "Regulator", "Client", "Site", "Standard"],
        "relation_types": ["DESIGNED_BY", "BUILT_BY", "USES_MATERIAL", "APPROVED_BY",
                           "COMMISSIONED_BY", "LOCATED_ON", "COMPLIES_WITH", "INSPECTED_BY"],
        "style_hints": "planning applications, construction contracts, inspection reports, architect briefs"
    },
]

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise data generation assistant. "
    "You generate realistic text snippets and extract entities and relations from them. "
    "Always follow the output format exactly — no extra commentary."
)

def build_prompt(domain: dict, variation_seed: int) -> str:
    entity_types = ", ".join(domain["entity_types"])
    relation_types = ", ".join(domain["relation_types"])

    # Vary complexity and style to improve diversity
    complexity = ["simple (2-3 sentences)", "moderate (3-4 sentences)", "detailed (4-5 sentences)"]
    comp = complexity[variation_seed % 3]
    perspectives = [
        "written as an internal memo",
        "written as a formal report excerpt",
        "written as informal notes",
        "written as a third-person description",
        "written as a summary paragraph",
    ]
    persp = perspectives[variation_seed % len(perspectives)]

    return f"""Domain: {domain["description"]}
Style: {domain["style_hints"]}
Length: {comp}, {persp}

Generate a realistic unstructured text snippet for this domain, then extract all entities and relations.

OUTPUT FORMAT (follow exactly):
TEXT: <your generated text here>
---
ENTITY: <name> | <type>
ENTITY: <name> | <type>
...
RELATION: <entity name> | <relation type> | <entity name>
RELATION: <entity name> | <relation type> | <entity name>
...

Rules:
- Entity names must appear verbatim (or as clear substrings) in the TEXT
- Each RELATION must use only entities listed above it
- Generate between 3 and 8 entities
- Generate between 2 and 6 relations
- Available entity types: {entity_types}
- Available relation types: {relation_types}
- Do not include any text outside the format above
"""

# ── Ollama API ────────────────────────────────────────────────────────────────

def ollama_generate(model: str, prompt: str, temperature: float = 0.75) -> Optional[str]:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 600,
                    "top_p": 0.9,
                },
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        print(f"\n  [ollama error] {e}", file=sys.stderr)
        return None


def check_model_available(model: str) -> bool:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        names = [m["name"] for m in resp.json().get("models", [])]
        return any(n == model or n.startswith(model.split(":")[0]) for n in names)
    except Exception:
        return False


def pull_model(model: str):
    print(f"Pulling {model} — this may take a few minutes...")
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model, "stream": False},
            timeout=600,
        )
        resp.raise_for_status()
        print(f"Pulled {model} successfully.")
    except Exception as e:
        print(f"Pull failed: {e}", file=sys.stderr)
        sys.exit(1)

# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_response(text: str, domain: dict) -> Optional[dict]:
    """Parse model output into a structured training record. Returns None if invalid."""
    if "---" not in text:
        return None

    parts = text.split("---", 1)
    text_part   = parts[0].strip()
    struct_part = parts[1].strip()

    # Extract TEXT:
    if not text_part.startswith("TEXT:"):
        return None
    input_text = text_part[5:].strip()
    if len(input_text) < 30:
        return None

    valid_entity_types   = set(domain["entity_types"])
    valid_relation_types = set(domain["relation_types"])

    entities  = []
    relations = []

    for line in struct_part.splitlines():
        line = line.strip()
        if line.startswith("ENTITY:"):
            parts_ = line[7:].split("|", 1)
            if len(parts_) == 2:
                name  = parts_[0].strip()
                etype = parts_[1].strip()
                if name and etype in valid_entity_types:
                    entities.append({"name": name, "type": etype})

        elif line.startswith("RELATION:"):
            parts_ = line[9:].split("|")
            if len(parts_) == 3:
                e1  = parts_[0].strip()
                rel = parts_[1].strip()
                e2  = parts_[2].strip()
                if e1 and rel and e2 and rel in valid_relation_types:
                    relations.append({"from": e1, "type": rel, "to": e2})

    if len(entities) < 2 or len(relations) < 1:
        return None

    # Cross-validate: relation endpoints must be listed entities
    entity_names = {e["name"] for e in entities}
    valid_relations = [r for r in relations
                       if r["from"] in entity_names and r["to"] in entity_names]
    if not valid_relations:
        return None

    # Build the canonical training pair strings
    output_lines = (
        [f"ENTITY: {e['name']} | {e['type']}" for e in entities] +
        [f"RELATION: {r['from']} | {r['type']} | {r['to']}" for r in valid_relations]
    )

    return {
        "input":          f"Extract entities and relations:\n{input_text}",
        "output":         "\n".join(output_lines),
        "domain":         domain["name"],
        "entity_count":   len(entities),
        "relation_count": len(valid_relations),
        "source":         "synthetic_ollama",
    }

# ── Generation loop ───────────────────────────────────────────────────────────

def generate_one(domain: dict, model: str, seed: int) -> Optional[dict]:
    """Generate and parse a single example. Returns None on failure."""
    prompt = build_prompt(domain, variation_seed=seed)
    for attempt in range(MAX_RETRIES):
        raw = ollama_generate(model, prompt, temperature=0.7 + random.uniform(0, 0.2))
        if raw:
            result = parse_response(raw, domain)
            if result:
                return result
        if attempt < MAX_RETRIES - 1:
            time.sleep(0.5)
    return None


def generate_domain(domain: dict, model: str, n: int, outfile,
                    start_idx: int = 0, workers: int = 1) -> int:
    """Generate n examples for a domain, writing to outfile. Returns count generated."""
    write_lock = threading.Lock()
    success    = 0
    fail       = 0

    seeds = [start_idx + i + random.randint(0, 50) for i in range(n * 3)]  # extra seeds for retries

    with tqdm(total=n, desc=f"  {domain['name']}", unit="ex", leave=True) as bar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(generate_one, domain, model, s): s for s in seeds[:n + 20]}

            for fut in as_completed(futures):
                if success >= n:
                    # Cancel remaining futures once target hit
                    for f in futures:
                        f.cancel()
                    break

                result = fut.result()
                if result:
                    with write_lock:
                        outfile.write(json.dumps(result) + "\n")
                        outfile.flush()
                    success += 1
                    bar.update(1)
                    bar.set_postfix(fails=fail)
                else:
                    fail += 1
                    bar.set_postfix(fails=fail)
                    if fail > n * 3:
                        tqdm.write(f"  [warn] too many failures in {domain['name']}, stopping early")
                        break

    return success


def load_existing(path: Path) -> dict[str, int]:
    """Return count of already-generated examples per domain."""
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                d = rec.get("domain", "")
                counts[d] = counts.get(d, 0) + 1
            except json.JSONDecodeError:
                pass
    return counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ER extraction training data")
    parser.add_argument("--model",      default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--per-domain", type=int, default=N_PER_DOMAIN,
                        help="Examples to generate per domain")
    parser.add_argument("--output",     default=OUTPUT_FILE, help="Output JSONL path")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip domains already at target count")
    parser.add_argument("--domains",    nargs="*", help="Only generate for these domain names")
    parser.add_argument("--workers",    type=int, default=3,
                        help="Concurrent Ollama requests (default: 3)")
    args = parser.parse_args()

    output_path = Path(__file__).parent / args.output

    # Check model
    if not check_model_available(args.model):
        print(f"Model '{args.model}' not found locally.")
        ans = input(f"Pull it now? (~4-5GB for 7B) [y/N]: ").strip().lower()
        if ans == "y":
            pull_model(args.model)
        else:
            print("Aborted. Run: ollama pull qwen2.5:7b")
            sys.exit(1)

    # Filter domains
    domains = DOMAINS
    if args.domains:
        domains = [d for d in DOMAINS if d["name"] in args.domains]
        if not domains:
            print(f"No matching domains found. Available: {[d['name'] for d in DOMAINS]}")
            sys.exit(1)

    # Resume state
    existing = load_existing(output_path) if args.resume else {}

    total_target = len(domains) * args.per_domain
    print(f"Generating {args.per_domain} examples × {len(domains)} domains = {total_target} total")
    print(f"Model: {args.model}  |  Workers: {args.workers}  |  Output: {output_path}")
    if args.resume and existing:
        print(f"Resuming — already have: {existing}")
    print()

    total_generated = 0
    with open(output_path, "a") as f:
        for domain in domains:
            have = existing.get(domain["name"], 0)
            need = args.per_domain - have
            if need <= 0:
                print(f"  [{domain['name']}] already complete ({have}/{args.per_domain}), skipping")
                total_generated += have
                continue

            print(f"  [{domain['name']}] generating {need} examples...")
            n = generate_domain(domain, args.model, need, f, start_idx=have,
                                workers=args.workers)
            total_generated += have + n

    print(f"\nDone. Total records in {output_path}: ", end="")
    with open(output_path) as f:
        print(sum(1 for _ in f))


if __name__ == "__main__":
    main()
