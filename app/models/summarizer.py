"""Summarization model orchestration."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import monotonic
from typing import Any, Dict, Tuple

from flask import current_app
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.exceptions import SummaryTimeoutError

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelResources:
    """Tokenizer/model bundle for one summarization model."""

    tokenizer: Any
    model: Any


class MultiModelSummarizer:
    """Lazy-loading multi-model summarization pipeline."""

    def __init__(self) -> None:
        """Initialize model cache and configuration."""
        self._lock = threading.Lock()
        self._resources: Dict[str, ModelResources] = {}

    def _model_config(self) -> Dict[str, str]:
        """Get configured model names.

        Returns:
            Mapping from summary key to HuggingFace model id.
        """
        return {
            "summary_bart": current_app.config["BART_MODEL_NAME"],
            "summary_pegasus": current_app.config["PEGASUS_MODEL_NAME"],
            "summary_t5": current_app.config["T5_MODEL_NAME"],
        }

    def _get_resources(self, summary_key: str) -> ModelResources:
        """Load tokenizer/model for one model key on first use.

        Args:
            summary_key: Internal summary identifier.

        Returns:
            Loaded model resources.
        """
        if summary_key in self._resources:
            return self._resources[summary_key]

        with self._lock:
            if summary_key in self._resources:
                return self._resources[summary_key]

            model_name = self._model_config()[summary_key]
            LOGGER.info("Loading model resources for %s (%s)", summary_key, model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._resources[summary_key] = ModelResources(tokenizer=tokenizer, model=model)
            return self._resources[summary_key]

    def _generate_with_model(self, summary_key: str, text: str) -> str:
        """Generate one summary with a specific model.

        Args:
            summary_key: One of summary_bart, summary_pegasus, summary_t5.
            text: Source text.

        Returns:
            Generated summary text.
        """
        resources = self._get_resources(summary_key)
        tokenizer = resources.tokenizer
        model = resources.model

        input_text = f"summarize: {text}" if summary_key == "summary_t5" else text
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=current_app.config["MAX_INPUT_TOKENS"],
        )
        generated = model.generate(
            **inputs,
            max_length=current_app.config["SUMMARY_MAX_LENGTH"],
            min_length=current_app.config["SUMMARY_MIN_LENGTH"],
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
        output = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return output

    def should_use_parallel(self) -> bool:
        """Decide whether summary generation should run in parallel.

        Returns:
            True when configured to use parallel mode in the current device context.
        """
        is_gpu = torch.cuda.is_available()
        if is_gpu:
            return bool(current_app.config["USE_PARALLEL_ON_GPU"])
        return bool(current_app.config["USE_PARALLEL_ON_CPU"])

    def _generate_parallel(self, text: str, timeout_seconds: int) -> Dict[str, str]:
        """Generate summaries in parallel.

        Args:
            text: Source text to summarize.
            timeout_seconds: Maximum total time for all models.

        Returns:
            Mapping with summary_bart, summary_pegasus, summary_t5.

        Raises:
            SummaryTimeoutError: If generation exceeds timeout.
        """
        model_keys = tuple(self._model_config().keys())
        summaries: Dict[str, str] = {}
        start_time = monotonic()
        futures: Dict[Future[str], str] = {}

        LOGGER.info("Starting parallel summary generation")
        with ThreadPoolExecutor(max_workers=len(model_keys)) as executor:
            for model_key in model_keys:
                futures[executor.submit(self._generate_with_model, model_key, text)] = model_key

            try:
                for future in as_completed(futures, timeout=timeout_seconds):
                    model_key = futures[future]
                    summaries[model_key] = future.result()
                    LOGGER.info("Completed summary for %s", model_key)
            except TimeoutError as error:
                for future in futures:
                    future.cancel()
                elapsed = monotonic() - start_time
                raise SummaryTimeoutError(
                    f"Model summarization timed out after {elapsed:.1f}s. "
                    "Please try a shorter document."
                ) from error

        elapsed = monotonic() - start_time
        if len(summaries) != len(model_keys):
            missing = [key for key in model_keys if key not in summaries]
            raise RuntimeError(f"Summary generation failed for: {', '.join(missing)}")

        if elapsed > timeout_seconds:
            raise SummaryTimeoutError(
                f"Model summarization timed out after {elapsed:.1f}s. "
                "Please try a shorter document."
            )

        LOGGER.info("All summaries generated in %.2fs", elapsed)
        return summaries

    def _generate_sequential(self, text: str, timeout_seconds: int) -> Dict[str, str]:
        """Generate summaries sequentially.

        Args:
            text: Source text to summarize.
            timeout_seconds: Maximum total time across all models.

        Returns:
            Mapping with summary_bart, summary_pegasus, summary_t5.

        Raises:
            SummaryTimeoutError: If total time exceeds timeout.
        """
        model_keys = tuple(self._model_config().keys())
        summaries: Dict[str, str] = {}
        start_time = monotonic()

        LOGGER.info("Starting sequential summary generation")
        for model_key in model_keys:
            elapsed = monotonic() - start_time
            if elapsed > timeout_seconds:
                raise SummaryTimeoutError(
                    f"Model summarization timed out after {elapsed:.1f}s. "
                    "Please try a shorter document."
                )
            summaries[model_key] = self._generate_with_model(model_key, text)
            LOGGER.info("Completed summary for %s", model_key)

        elapsed = monotonic() - start_time
        if elapsed > timeout_seconds:
            raise SummaryTimeoutError(
                f"Model summarization timed out after {elapsed:.1f}s. "
                "Please try a shorter document."
            )

        LOGGER.info("All summaries generated in %.2fs", elapsed)
        return summaries

    def generate_summaries(
        self,
        text: str,
        timeout_seconds: int,
        use_parallel: bool = False,
    ) -> Dict[str, str]:
        """Generate summaries with CPU/GPU-aware execution mode.

        Args:
            text: Source text to summarize.
            timeout_seconds: Maximum total time for all models.
            use_parallel: False for CPU-oriented sequential execution,
                True for GPU-oriented parallel execution.

        Returns:
            Mapping with summary_bart, summary_pegasus, summary_t5.
        """
        if use_parallel:
            return self._generate_parallel(text=text, timeout_seconds=timeout_seconds)
        return self._generate_sequential(text=text, timeout_seconds=timeout_seconds)

    @staticmethod
    def choose_routed_model(document_type: str) -> Tuple[str, str]:
        """Choose best-fit model by document type.

        Args:
            document_type: Detected or user-specified document type.

        Returns:
            Tuple of (summary_key, model_label).
        """
        model_by_type = {
            "research_paper": ("summary_pegasus", "pegasus"),
            "announcement": ("summary_t5", "t5"),
            "news": ("summary_bart", "bart"),
        }
        return model_by_type.get(document_type, ("summary_bart", "bart"))
