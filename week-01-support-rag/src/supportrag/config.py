"""Settings, read from the environment.

Everything that changes between machines or experiments lives here, so the
rest of the code never calls os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader: KEY=VALUE lines, existing variables win.

    Ten lines is cheaper than a dependency. It does not handle quoting or
    multi-line values; use real environment variables if you need those.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    chat_model: str
    embedding_model: str
    # Retrieval
    top_k: int = 5
    candidates_per_retriever: int = 20
    # Agent
    max_steps: int = 5

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.sqlite"


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        data_dir=Path(os.environ.get("SUPPORTRAG_DATA_DIR", "data")),
        # The cheapest current model is the default so experiments cost
        # fractions of a cent. Raise it once the evals say you need to.
        chat_model=os.environ.get("SUPPORTRAG_CHAT_MODEL", "gpt-6-luna"),
        embedding_model=os.environ.get("SUPPORTRAG_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
