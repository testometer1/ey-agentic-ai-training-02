"""Build ~200 support-ticket text files (issue + resolution + metadata)."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
TICKET_DIR = os.path.join(HERE, "tickets")

LOGIN_ISSUE = (
    "The customer reported that the session expires randomly every few minutes "
    "during a live retrieval demo."
)
LOGIN_RES = (
    "Agent reset the refresh-token TTL from 3 minutes to 45 minutes on the auth service, "
    "cleared the stuck session on device id D-4412, and confirmed the console stayed authenticated."
)

TEMPLATES = {
    "login_session": [
        (
            "User is returned to the credential prompt while the agent loop is still running.",
            "Increased session lifetime and documented the auth TTL change in the runbook.",
        ),
        (
            "Mobile token dies mid-call so identity is dropped during the retrieval demo.",
            "Patched silent token refresh; customer confirmed they were not asked for a password again.",
        ),
        (
            "Console identity drops while viewing embeddings in the support console.",
            "Fixed cookie Secure flag mismatch between London and Dublin gateways.",
        ),
    ],
    "agent_loop": [
        (
            "The live agent entered a tight tool call loop after a timeout.",
            "Guardrail TOOL-GRD-3 stopped identical retries after three calls; human took over.",
        ),
        (
            "Planner skipped observe and retried the billing API too quickly.",
            "Forced observe-before-act in the loop config; no funds were moved.",
        ),
        (
            "Control loop planned then acted then observed the new error state too late.",
            "Reordered the cycle and added a Splunk alert for repeated tool names.",
        ),
    ],
    "retrieval": [
        (
            "Vector database returned no policy chunks so the model invented vacation days.",
            "Rebuilt index EMB-IDX-EU-9 and added a no-chunk refusal in the system prompt.",
        ),
        (
            "Embeddings for European tickets were stale after the March policy change.",
            "Re-ingested POL-HR extracts and verified cosine rank on a paraphrase query.",
        ),
        (
            "Chroma collection pointed at the wrong persist directory in Singapore.",
            "Relinked the collection and re-embedded 200 ticket files.",
        ),
    ],
    "risk": [
        (
            "Conservative client Sarah Chen was shown an aggressive trade ticket by mistake.",
            "Risk desk reaffirmed RSK-PROF-C01 conservative; trade was not sent.",
        ),
        (
            "Profile review missing after a large house-deposit wire on ACC-4412.",
            "RM logged the 22 July review; profile stayed conservative.",
        ),
        (
            "Model promised a return figure that was not on the July activity extract.",
            "Fact-check script blocked send; RM rewrote the two-paragraph review.",
        ),
    ],
    "tools": [
        (
            "Production executor selected an unsafe tool against the live billing API.",
            "Change TOOL-GRD-3 rejected the call; CloudWatch alarm cleared after audit capture was on.",
        ),
        (
            "Identical production tool call repeated more than three times.",
            "Guardrail blocked the loop; operator in London confirmed no customer impact.",
        ),
        (
            "Unsafe tool use on the embedding rebuild job.",
            "Job rerun with read-only credentials; index rebuilt without dropping prod.",
        ),
    ],
}

CUSTOMERS = [f"CUST-{n}" for n in range(4400, 4480)] + [f"CUST-{n}" for n in range(8800, 8840)]
# 4021's customer and related login customers
SPECIAL = {
    4021: ("CUST-4412", "login_session", LOGIN_ISSUE, LOGIN_RES, "resolved"),
    4022: (
        "CUST-8801",
        "login_session",
        "Customer cannot stay authenticated; the console drops the session every few minutes during ticket search.",
        "Same TTL fix as TCK-4021; customer later closed the account.",
        "resolved",
    ),
    4023: (
        "CUST-2190",
        "login_session",
        "Identity drops from the console while waiting on retrieval answers.",
        "Applied the refresh-token patch from TCK-4021; still an active customer.",
        "resolved",
    ),
    4100: (
        "CUST-9901",
        "retrieval",
        "Policy RAG answered a gym-membership question that is not in any HR document.",
        "Tightened the grounded prompt so out-of-scope questions return I don't know.",
        "resolved",
    ),
}


def ticket_body(tid, customer, category, issue, resolution, status):
    return (
        f"TICKET: TCK-{tid}\n"
        f"CUSTOMER: {customer}\n"
        f"CATEGORY: {category}\n"
        f"STATUS: {status}\n"
        f"QUEUE: {category}\n\n"
        f"Customer:\n{issue}\n\n"
        f"Agent:\nAcknowledged the {category} queue item and checked related runs.\n\n"
        f"Resolution:\n{resolution}\n"
        f"STATUS_FINAL: {status}\n"
    )


def generate(n=200, start=4001):
    os.makedirs(TICKET_DIR, exist_ok=True)
    cats = list(TEMPLATES)
    written = []
    for i in range(n):
        tid = start + i
        if tid in SPECIAL:
            customer, category, issue, resolution, status = SPECIAL[tid]
        else:
            category = cats[i % len(cats)]
            issue, resolution = TEMPLATES[category][i % len(TEMPLATES[category])]
            issue = issue + f" Ticket sequence {tid}."
            customer = CUSTOMERS[i % len(CUSTOMERS)]
            status = "resolved" if i % 7 else "open"
        path = os.path.join(TICKET_DIR, f"TCK-{tid}.txt")
        open(path, "w", encoding="utf-8").write(ticket_body(tid, customer, category, issue, resolution, status))
        written.append(path)
    return written


def ensure_tickets():
    generate()
    return TICKET_DIR
