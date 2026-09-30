"""Two read only MCP tools backed by tiny, fictional JSON fixtures."""

import json
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("medical-demo")
DATA = Path(__file__).parent / "data"


@mcp.tool()
def get_patient(user_id: str) -> dict[str, Any]:
    """Get one fictional patient's basic information and known conditions."""
    patients = json.loads((DATA / "patients.json").read_text())
    patient = patients.get(user_id)
    return {"found": patient is not None, "user_id": user_id, "patient": patient}


@mcp.tool()
def search_umls(query: str) -> dict[str, Any]:
    """Match names or aliases in a synthetic UMLS-like terminology fixture."""
    concepts = json.loads((DATA / "umls.json").read_text())
    matches = [
        concept for concept in concepts
        if any(
            re.search(r"\b" + re.escape(term) + r"\b", query, re.IGNORECASE)
            for term in [concept["name"], *concept["aliases"]]
        )
    ]
    return {"matches": matches, "source": "Synthetic fixture, not official UMLS"}


if __name__ == "__main__":
    # stdout belongs to the MCP protocol. Do not print application logs here.
    mcp.run(transport="stdio")
