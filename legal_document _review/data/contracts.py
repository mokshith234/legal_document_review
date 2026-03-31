"""
data/contracts.py
-----------------
Ground-truth contract data used across all 3 tasks.

WHY THIS FILE EXISTS:
  The environment needs realistic, varied contract text to serve agents.
  Ground truth labels here are what graders compare agent outputs against.
  Having this in one place means tasks share consistent data —
  if we update a clause, all graders automatically use the new version.

  In a production system this would be a database. For the hackathon,
  a structured Python dict gives us speed + reproducibility.
"""

from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# TASK 1 DATA — Clause Classification
# Taxonomy of 8 clause types. Agent must identify which type a clause is.
# ---------------------------------------------------------------------------

CLAUSE_TAXONOMY = [
    "indemnification",
    "limitation_of_liability",
    "confidentiality",
    "termination",
    "intellectual_property",
    "governing_law",
    "force_majeure",
    "payment_terms",
]

# Near-misses for partial credit scoring
CLAUSE_NEAR_MISSES: Dict[str, List[str]] = {
    "indemnification":        ["limitation_of_liability"],
    "limitation_of_liability":["indemnification"],
    "confidentiality":        ["intellectual_property"],
    "intellectual_property":  ["confidentiality"],
    "termination":            ["payment_terms"],
    "governing_law":          ["force_majeure"],
    "force_majeure":          ["governing_law"],
    "payment_terms":          ["termination"],
}

CLASSIFICATION_SAMPLES: List[Dict[str, Any]] = [
    {
        "id": "cls_001",
        "clause": (
            "Each party shall defend, indemnify, and hold harmless the other party "
            "from and against any claims, damages, losses, and expenses, including "
            "reasonable attorneys' fees, arising out of or relating to the indemnifying "
            "party's breach of this Agreement."
        ),
        "label": "indemnification",
        "difficulty": "easy",
    },
    {
        "id": "cls_002",
        "clause": (
            "In no event shall either party be liable to the other for any indirect, "
            "incidental, special, exemplary, or consequential damages, regardless of "
            "whether such party has been advised of the possibility of such damages. "
            "Each party's total cumulative liability shall not exceed the fees paid "
            "in the twelve months preceding the claim."
        ),
        "label": "limitation_of_liability",
        "difficulty": "easy",
    },
    {
        "id": "cls_003",
        "clause": (
            "The Receiving Party agrees to hold the Disclosing Party's Confidential "
            "Information in strict confidence and to not disclose such information to "
            "any third party without prior written consent. This obligation shall "
            "survive termination of the Agreement for a period of five (5) years."
        ),
        "label": "confidentiality",
        "difficulty": "easy",
    },
    {
        "id": "cls_004",
        "clause": (
            "Either party may terminate this Agreement for convenience upon thirty "
            "(30) days written notice. Company may terminate immediately upon written "
            "notice if Vendor materially breaches any provision of this Agreement and "
            "fails to cure such breach within fifteen (15) days of notice."
        ),
        "label": "termination",
        "difficulty": "medium",
    },
    {
        "id": "cls_005",
        "clause": (
            "All inventions, works of authorship, and other intellectual property "
            "created by Vendor in connection with the Services shall be considered "
            "works made for hire. To the extent any such work does not qualify as "
            "work made for hire, Vendor hereby irrevocably assigns all right, title, "
            "and interest therein to Company."
        ),
        "label": "intellectual_property",
        "difficulty": "medium",
    },
    {
        "id": "cls_006",
        "clause": (
            "This Agreement shall be governed by and construed in accordance with "
            "the laws of the State of Delaware, without regard to its conflict of "
            "law provisions. Any disputes shall be resolved exclusively in the "
            "state or federal courts located in Wilmington, Delaware."
        ),
        "label": "governing_law",
        "difficulty": "easy",
    },
    {
        "id": "cls_007",
        "clause": (
            "Neither party shall be liable for any delay or failure to perform its "
            "obligations under this Agreement to the extent caused by circumstances "
            "beyond its reasonable control, including acts of God, war, terrorism, "
            "pandemic, labor disputes, or governmental actions, provided that the "
            "affected party provides prompt written notice to the other party."
        ),
        "label": "force_majeure",
        "difficulty": "medium",
    },
    {
        "id": "cls_008",
        "clause": (
            "Client shall pay all undisputed invoices within thirty (30) days of "
            "receipt. Overdue payments shall accrue interest at the rate of 1.5% per "
            "month or the maximum rate permitted by law, whichever is less. Vendor "
            "reserves the right to suspend services upon sixty (60) days of non-payment."
        ),
        "label": "payment_terms",
        "difficulty": "easy",
    },
    {
        "id": "cls_009",
        "clause": (
            "The provisions of Sections 4 (Confidentiality), 6 (Intellectual Property), "
            "8 (Indemnification), and 9 (Limitation of Liability) shall survive the "
            "expiration or termination of this Agreement, together with any other "
            "provisions that by their nature should survive."
        ),
        "label": "termination",
        "difficulty": "hard",
    },
    {
        "id": "cls_010",
        "clause": (
            "Vendor shall not subcontract or assign any rights or obligations under "
            "this Agreement without Company's prior written consent. Any purported "
            "assignment without such consent shall be null and void. Notwithstanding "
            "the foregoing, either party may assign this Agreement in its entirety "
            "in connection with a merger, acquisition, or sale of all or substantially "
            "all of its assets."
        ),
        "label": "intellectual_property",
        "difficulty": "hard",
    },
]


# ---------------------------------------------------------------------------
# TASK 2 DATA — Risk Spotting
# Full contract sections with annotated risks agents must find.
# ---------------------------------------------------------------------------

RISK_SAMPLES: List[Dict[str, Any]] = [
    {
        "id": "risk_001",
        "section_title": "Software Development Services Agreement — Liability & Indemnification",
        "contract_text": (
            "8. INDEMNIFICATION. Client shall indemnify, defend, and hold harmless "
            "Vendor from and against any and all claims arising from Client's use of "
            "the Software. Vendor's indemnification obligations are limited solely to "
            "claims of intellectual property infringement by the Software as delivered.\n\n"
            "9. LIMITATION OF LIABILITY. VENDOR'S TOTAL LIABILITY SHALL NOT EXCEED "
            "ONE HUNDRED DOLLARS ($100). THIS LIMITATION APPLIES TO ALL CLAIMS "
            "INCLUDING BREACH OF CONTRACT, TORT, AND NEGLIGENCE. VENDOR SHALL NOT "
            "BE LIABLE FOR DATA LOSS UNDER ANY CIRCUMSTANCES.\n\n"
            "10. AUDIT RIGHTS. Company grants Vendor the right to audit Client's "
            "systems and data at any time with 24 hours notice to verify compliance. "
            "Client must provide Vendor with full access to all internal systems, "
            "databases, and confidential business records upon request."
        ),
        "ground_truth_risks": [
            {
                "risk_id": "r1",
                "clause_ref": "Section 8",
                "risk": "One-sided indemnification — Client bears all indemnification but Vendor has minimal obligations",
                "severity": "high",
            },
            {
                "risk_id": "r2",
                "clause_ref": "Section 9",
                "risk": "Liability cap of $100 is unreasonably low for a software contract — effectively zero protection",
                "severity": "critical",
            },
            {
                "risk_id": "r3",
                "clause_ref": "Section 9",
                "risk": "Blanket exclusion of data loss liability is dangerous — vendor has no accountability for data breaches",
                "severity": "critical",
            },
            {
                "risk_id": "r4",
                "clause_ref": "Section 10",
                "risk": "Vendor audit rights are excessively broad — access to all internal systems with minimal notice is a security risk",
                "severity": "high",
            },
        ],
        "difficulty": "medium",
    },
    {
        "id": "risk_002",
        "section_title": "SaaS Subscription Agreement — Data & IP",
        "contract_text": (
            "5. DATA OWNERSHIP. All data uploaded to the Platform by Client becomes "
            "the property of Vendor upon upload. Vendor may use, analyze, share, and "
            "sell aggregated and non-aggregated Client data for any purpose, including "
            "training machine learning models and commercial resale to third parties.\n\n"
            "6. INTELLECTUAL PROPERTY. Any improvements, modifications, or derivative "
            "works of the Platform suggested by Client, whether implemented by Vendor "
            "or not, shall become the sole property of Vendor with no compensation "
            "owed to Client.\n\n"
            "7. RENEWAL. This Agreement automatically renews for successive one-year "
            "terms. Client must provide written notice of non-renewal at least 180 days "
            "before the end of any term. Failure to provide timely notice results in "
            "a renewal fee equal to 150% of the prior year's fees."
        ),
        "ground_truth_risks": [
            {
                "risk_id": "r1",
                "clause_ref": "Section 5",
                "risk": "Vendor claims ownership of all uploaded client data — extremely unusual and dangerous",
                "severity": "critical",
            },
            {
                "risk_id": "r2",
                "clause_ref": "Section 5",
                "risk": "Vendor can sell non-aggregated client data to third parties — major privacy and confidentiality violation",
                "severity": "critical",
            },
            {
                "risk_id": "r3",
                "clause_ref": "Section 6",
                "risk": "Client suggestions become Vendor IP with no compensation — disincentivizes client feedback and is unfair",
                "severity": "high",
            },
            {
                "risk_id": "r4",
                "clause_ref": "Section 7",
                "risk": "180-day non-renewal notice is unusually long — creates vendor lock-in and auto-renewal trap",
                "severity": "medium",
            },
            {
                "risk_id": "r5",
                "clause_ref": "Section 7",
                "risk": "150% renewal penalty fee is punitive and non-standard",
                "severity": "medium",
            },
        ],
        "difficulty": "hard",
    },
]


# ---------------------------------------------------------------------------
# TASK 3 DATA — Contract Redlining
# Full contract + policy brief. Agent must propose specific edits.
# ---------------------------------------------------------------------------

REDLINE_SAMPLES: List[Dict[str, Any]] = [
    {
        "id": "redline_001",
        "contract_title": "Master Services Agreement — Version 1.0 (Vendor Draft)",
        "contract_text": (
            "1. SERVICES. Vendor shall provide software development services as "
            "described in Statements of Work. Vendor retains sole discretion to "
            "determine the manner and means of performing the Services.\n\n"
            "2. FEES. Client shall pay Vendor's invoices within 7 days of receipt. "
            "Late payments accrue interest at 5% per month. Vendor may immediately "
            "suspend services upon any late payment without notice.\n\n"
            "3. IP OWNERSHIP. All work product, inventions, and deliverables created "
            "under this Agreement shall be owned solely by Vendor. Client receives "
            "a non-exclusive, non-transferable license to use deliverables solely "
            "for internal purposes.\n\n"
            "4. WARRANTY DISCLAIMER. ALL SERVICES AND DELIVERABLES ARE PROVIDED "
            "AS-IS. VENDOR MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING "
            "FITNESS FOR A PARTICULAR PURPOSE.\n\n"
            "5. CONFIDENTIALITY. Each party agrees to keep confidential information "
            "of the other party confidential for one (1) year from disclosure.\n\n"
            "6. GOVERNING LAW. This Agreement is governed by the laws of the Cayman "
            "Islands. All disputes shall be resolved by binding arbitration in the "
            "Cayman Islands."
        ),
        "policy_brief": (
            "Our standard contract policy requires:\n"
            "- Payment terms: Net 30 (not Net 7)\n"
            "- Late interest: Maximum 1.5% per month\n"
            "- Service suspension: Requires 30 days written notice\n"
            "- IP ownership: Work product must be owned by Client (work-for-hire)\n"
            "- Confidentiality: Minimum 3-year obligation\n"
            "- Governing law: Must be US jurisdiction (Delaware preferred)\n"
            "- Warranties: Vendor must warrant Services meet specifications for 90 days\n"
            "- Disputes: Arbitration permitted but must be in US, AAA rules"
        ),
        "ground_truth_redlines": [
            {
                "section": "Section 2 - Payment",
                "issue": "Net 7 payment terms too aggressive",
                "original": "Client shall pay Vendor's invoices within 7 days of receipt.",
                "redline": "Client shall pay Vendor's invoices within thirty (30) days of receipt.",
            },
            {
                "section": "Section 2 - Late Interest",
                "issue": "5%/month interest is usurious",
                "original": "Late payments accrue interest at 5% per month.",
                "redline": "Late payments shall accrue interest at 1.5% per month or the maximum permitted by applicable law, whichever is less.",
            },
            {
                "section": "Section 2 - Suspension",
                "issue": "Immediate suspension without notice is unfair",
                "original": "Vendor may immediately suspend services upon any late payment without notice.",
                "redline": "Vendor may suspend services upon thirty (30) days written notice if an undisputed invoice remains unpaid.",
            },
            {
                "section": "Section 3 - IP Ownership",
                "issue": "Vendor retaining all IP is non-standard — must flip to Client",
                "original": "All work product, inventions, and deliverables created under this Agreement shall be owned solely by Vendor.",
                "redline": "All work product, inventions, and deliverables created by Vendor in connection with the Services shall be works made for hire owned solely by Client. To the extent any deliverable does not qualify as work made for hire, Vendor hereby assigns all rights therein to Client.",
            },
            {
                "section": "Section 4 - Warranty",
                "issue": "Complete warranty disclaimer — vendor should warrant workmanship",
                "original": "ALL SERVICES AND DELIVERABLES ARE PROVIDED AS-IS. VENDOR MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING FITNESS FOR A PARTICULAR PURPOSE.",
                "redline": "Vendor warrants that Services will be performed in a professional manner and that Deliverables will materially conform to the applicable Statement of Work for ninety (90) days following delivery.",
            },
            {
                "section": "Section 5 - Confidentiality",
                "issue": "1-year confidentiality is too short",
                "original": "Each party agrees to keep confidential information of the other party confidential for one (1) year from disclosure.",
                "redline": "Each party agrees to maintain the confidentiality of the other party's Confidential Information for three (3) years from the date of disclosure.",
            },
            {
                "section": "Section 6 - Governing Law",
                "issue": "Cayman Islands jurisdiction is inappropriate for US contract",
                "original": "This Agreement is governed by the laws of the Cayman Islands. All disputes shall be resolved by binding arbitration in the Cayman Islands.",
                "redline": "This Agreement is governed by the laws of the State of Delaware. Disputes shall be resolved by binding arbitration in Wilmington, Delaware under the rules of the American Arbitration Association.",
            },
        ],
        "difficulty": "hard",
    },
]
