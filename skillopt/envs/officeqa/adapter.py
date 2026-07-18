from __future__ import annotations

import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.officeqa.dataloader import OfficeQADataLoader
from skillopt.envs.officeqa.rollout import _normalize_search_mode, run_batch
from skillopt.envs.officeqa.tool_runtime import (
    build_oracle_parsed_pages_context,
    resolve_candidate_files,
    resolve_docs_roots,
)


class OfficeQAAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        workers: int = 8,
        analyst_workers: int = 8,
        seed: int = 42,
        limit: int = 0,
        max_tool_turns: int = 12,
        max_completion_tokens: int = 16384,
        search_mode: str = "offline",
        max_queries_per_turn: int = 4,
        search_api_url: str = os.environ.get("OFFICEQA_SEARCH_API_URL", "http://localhost:8080/search_tool/search"),
        search_auth_env: str = "OFFICEQA_CUSTOM_SEARCH_AUTH",
        search_provider: str = "duckduckgo",
        search_max_num_results: int = 4,
        search_timeout_seconds: int = 20,
        use_local_tools: bool = True,
        data_dirs: list[str] | str | None = None,
        docs_dirs: list[str] | str | None = None,
    ) -> None:
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.max_tool_turns = max_tool_turns
        self.max_completion_tokens = int(max_completion_tokens)
        self.search_mode = str(search_mode or "offline")
        self.max_queries_per_turn = int(max_queries_per_turn)
        self.search_api_url = str(search_api_url or "").strip()
        self.search_auth_env = str(search_auth_env or "OFFICEQA_CUSTOM_SEARCH_AUTH").strip()
        self.search_provider = str(search_provider or "duckduckgo").strip()
        self.search_max_num_results = int(search_max_num_results)
        self.search_timeout_seconds = int(search_timeout_seconds)
        self.use_local_tools = bool(use_local_tools)
        self.data_dirs = data_dirs if data_dirs is not None else docs_dirs
        self.dataloader = OfficeQADataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)
        self._preflight_offline_evidence()

    def _preflight_offline_evidence(self) -> None:
        """Fail before training if the configured local corpus cannot serve evidence.

        A directory existing is not sufficient: a wrong OfficeQA corpus version
        can contain files while none of the split's ``source_files`` /
        ``source_docs`` references resolve. Without this check every rollout can
        still run successfully but receive no evidence and score zero.
        """
        if _normalize_search_mode(self.search_mode) != "offline":
            return

        docs_roots = resolve_docs_roots(self.data_dirs)
        probe_items = (
            list(self.dataloader.train_items[:2])
            + list(self.dataloader.val_items[:1])
            + list(self.dataloader.test_items[:1])
        )
        if not probe_items:
            raise RuntimeError("OfficeQA preflight found no split items to validate.")

        resolved = 0
        for item in probe_items:
            source_files = item.get("source_files", [])
            candidate_files = (
                resolve_candidate_files(source_files, docs_roots)
                if source_files
                else []
            )
            oracle_context = build_oracle_parsed_pages_context(
                source_files,
                item.get("source_docs", []),
                docs_roots,
            )
            if candidate_files or oracle_context:
                resolved += 1

        print(
            "[OfficeQA preflight] "
            f"docs_roots={docs_roots} evidence_resolved={resolved}/{len(probe_items)}"
        )
        if resolved == 0:
            raise RuntimeError(
                "OfficeQA docs directory exists, but none of the sampled split references "
                "resolved to local document evidence. Check OFFICEQA_DOCS_DIR and ensure it "
                "matches this OfficeQA split before training."
            )

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        items: list[dict] = env_manager
        return run_batch(
            items=items,
            out_root=out_dir,
            skill_content=skill_content,
            workers=self.workers,
            max_tool_turns=self.max_tool_turns,
            max_completion_tokens=self.max_completion_tokens,
            search_mode=self.search_mode,
            max_queries_per_turn=self.max_queries_per_turn,
            search_api_url=self.search_api_url,
            search_auth_env=self.search_auth_env,
            search_provider=self.search_provider,
            search_max_num_results=self.search_max_num_results,
            search_timeout_seconds=self.search_timeout_seconds,
            use_local_tools=self.use_local_tools,
            data_dirs=self.data_dirs,
            diagnostic_mode=kwargs.get("diagnostic_mode", False),
            diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
        )

    def get_task_types(self) -> list[str]:
        seen: list[str] = []
        for item in self.dataloader.train_items + self.dataloader.val_items + self.dataloader.test_items:
            task_type = str(item.get("task_type") or "officeqa")
            if task_type not in seen:
                seen.append(task_type)
        return seen or ["officeqa"]
