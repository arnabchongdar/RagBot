"""Synthetic insurance documents (report §5.1 "Both" corpus option).

Generates a small set of realistic ASCII-only insurance PDFs so the full
pipeline is runnable immediately. Content is written to cover the report's
three canonical queries (home water damage, motor claim filing, health claim
rejection). Drop your real PDFs in the docs dir to use them instead / as well.
"""

from __future__ import annotations

from pathlib import Path

# Each document: (filename, title, [(heading, body), ...])
_DOCUMENTS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "home_insurance_policy.pdf",
        "HomeShield Home Insurance - Policy Wording",
        [
            ("1. Insuring Agreement",
             "This policy covers the insured home structure and its contents against "
             "sudden, accidental, and unforeseen physical loss or damage, subject to the "
             "terms, conditions, limits, and exclusions set out below."),
            ("2. Covered Perils",
             "Covered perils include fire and lightning, storm and windstorm, theft and "
             "burglary following forcible entry, and escape of water. Escape of water means "
             "sudden and accidental discharge of water from a burst pipe, a ruptured water "
             "tank, or an overflowing household appliance such as a washing machine or "
             "dishwasher. Resulting damage to floors, walls, and contents is covered up to "
             "the sum insured."),
            ("3. Water Damage - Scope and Conditions",
             "Water damage that is sudden and accidental is covered. The insured must take "
             "reasonable steps to mitigate further damage, for example by shutting off the "
             "water supply. Coverage applies only where the property has been reasonably "
             "maintained."),
            ("4. Exclusions",
             "The following are excluded: (a) flood, storm surge, and rising ground water, "
             "unless the optional Flood Cover endorsement has been purchased; (b) gradual "
             "seepage, slow leaks, damp, or water ingress occurring over a period of time; "
             "(c) wear and tear, rust, corrosion, and lack of maintenance; (d) damage due "
             "to faulty workmanship or defective design. Water damage arising from a lack "
             "of maintenance or from a leak that developed gradually is not covered."),
            ("5. Deductible",
             "A standard deductible of Rs. 5,000 applies to each water damage claim. The "
             "deductible is the amount the insured bears before the insurer pays the balance."),
            ("6. Endorsements",
             "Optional endorsements include Flood Cover (extends the policy to flood and "
             "storm-surge water damage) and Escalation Cover. Endorsements are effective only "
             "when shown on the policy schedule and the additional premium has been paid."),
        ],
    ),
    (
        "motor_claims_manual.pdf",
        "DriveSafe Motor Insurance - Claims Procedure Manual",
        [
            ("1. Reporting an Accident Claim",
             "In the event of accidental damage to the insured vehicle, notify the insurer "
             "as soon as possible and in any case within 48 hours of the incident. Claims may "
             "be intimated through the 24x7 claims helpline, the mobile app, or the insurer "
             "web portal. Late intimation without valid reason may affect the claim."),
            ("2. Documents Required",
             "The following documents are required to file a motor own-damage claim: the "
             "completed and signed claim form; a copy of the insurance policy or the policy "
             "number; the vehicle Registration Certificate (RC); the driving licence of the "
             "person driving at the time of the accident; a police First Information Report "
             "(FIR) in cases of theft, third-party injury, or major accident; repair estimates "
             "from the garage; and photographs of the damage where available."),
            ("3. Survey and Inspection",
             "After intimation, the insurer assigns a surveyor to inspect the damaged vehicle "
             "and assess the loss. Repairs should not commence beyond dismantling until the "
             "surveyor has inspected the vehicle, unless the insurer authorizes otherwise."),
            ("4. Cashless vs Reimbursement",
             "At network garages the claim can be settled on a cashless basis, where the "
             "insurer pays the garage directly for the approved amount. At non-network garages "
             "the insured pays and then claims reimbursement by submitting paid invoices and "
             "payment receipts."),
            ("5. Deductible and Excess",
             "A compulsory excess (deductible) applies to every own-damage claim and is borne "
             "by the insured. Any voluntary excess opted for at purchase is additional. "
             "Depreciation on replaced parts may apply unless a zero-depreciation add-on was "
             "purchased."),
            ("6. Claim Settlement Timeline",
             "Once all required documents and the survey report are received, the insurer "
             "endeavours to approve the claim within 7 working days. The approved amount is "
             "paid to the network garage (cashless) or reimbursed to the insured."),
        ],
    ),
    (
        "health_insurance_policy.pdf",
        "CareFirst Health Insurance - Policy Terms and Claim Guidelines",
        [
            ("1. Coverage",
             "The policy covers medically necessary hospitalization expenses including room "
             "charges, ICU, surgeon and specialist fees, diagnostics, and prescribed "
             "medication, subject to the sum insured, sub-limits, and waiting periods."),
            ("2. Waiting Periods",
             "An initial waiting period of 30 days applies from the policy start date, during "
             "which only accidental hospitalization is covered. Pre-existing diseases are "
             "subject to a waiting period of 36 months of continuous coverage. Certain "
             "specified ailments such as cataract, hernia, and joint replacement have a "
             "24-month waiting period."),
            ("3. Sub-limits",
             "Room rent is capped at 1 percent of the sum insured per day unless a higher "
             "category is opted. Some treatments have specific monetary sub-limits. Amounts "
             "above the applicable sub-limit are borne by the insured."),
            ("4. Exclusions",
             "Permanent exclusions include cosmetic treatment, dental treatment unless arising "
             "from accident, self-inflicted injury, and treatment not medically necessary. "
             "Expenses outside the scope of hospitalization may not be payable."),
            ("5. Common Reasons a Claim Is Rejected",
             "A health insurance claim may be rejected or reduced for the following reasons: "
             "the treatment occurred during the initial 30-day waiting period; the condition "
             "is a pre-existing disease still within its 36-month waiting period; the ailment "
             "falls under a policy exclusion; there was non-disclosure or misrepresentation of "
             "material facts at the time of purchase; documents were missing or incomplete; or "
             "the claimed amount exceeded applicable sub-limits. The specific reason is stated "
             "in the claim rejection letter and depends on the policy terms and claim records."),
            ("6. Claim Documentation",
             "Cashless claims require pre-authorization at a network hospital. Reimbursement "
             "claims require the discharge summary, itemized hospital bills, payment receipts, "
             "diagnostic reports, and the doctor's prescription. Incomplete documentation is a "
             "common cause of delay or rejection."),
        ],
    ),
    (
        "insurance_faqs.pdf",
        "Insurance Frequently Asked Questions",
        [
            ("Q. What is the difference between a premium and a deductible?",
             "The premium is the amount you pay the insurer to keep the policy in force. The "
             "deductible (or excess) is the amount you bear yourself on a claim before the "
             "insurer pays the balance. A higher deductible usually means a lower premium."),
            ("Q. What is the difference between coverage and an exclusion?",
             "Coverage describes what the policy pays for. An exclusion is a specific situation "
             "or cause of loss that the policy does not pay for. Always read both the coverage "
             "and the exclusions to understand what is protected."),
            ("Q. What is a waiting period?",
             "A waiting period is a duration after the policy starts during which certain "
             "conditions or treatments are not yet covered. Common examples are the initial "
             "waiting period and the pre-existing disease waiting period in health insurance."),
            ("Q. How do I renew my policy and what is a grace period?",
             "Renew the policy before its expiry date to maintain continuous coverage and "
             "accrued waiting-period credits. A grace period, often 30 days, allows renewal "
             "after expiry, but losses occurring during the lapse are generally not covered."),
            ("Q. What is an endorsement?",
             "An endorsement is a written amendment that adds, removes, or changes coverage "
             "on the policy, such as adding flood cover or a zero-depreciation add-on. It is "
             "effective only when shown on the policy schedule."),
            ("Q. What is the sum insured?",
             "The sum insured is the maximum amount the insurer will pay under the policy "
             "during the policy period. Claims are settled up to this limit, subject to "
             "sub-limits, deductibles, and conditions."),
        ],
    ),
]


def _write_pdf(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(w=0, h=9, text=title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(w=0, h=7, text=heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(w=0, h=6, text=body, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    pdf.output(str(path))


def generate_samples(out_dir: Path, *, overwrite: bool = False) -> list[Path]:
    """Write the synthetic insurance PDFs into ``out_dir``. Returns written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, title, sections in _DOCUMENTS:
        path = out_dir / filename
        if path.exists() and not overwrite:
            written.append(path)
            continue
        _write_pdf(path, title, sections)
        written.append(path)
    return written
