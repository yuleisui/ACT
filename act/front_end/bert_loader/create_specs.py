#===- act/front_end/bert_loader/create_specs.py - BERT Specs ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Creates embedding-space InputSpec and OutputSpec pairs for SST/Yelp BERT
#   classifiers whose verification graph starts after token embedding lookup.
#
#===---------------------------------------------------------------------===#

"""Specification creator for BERT verify-from-embeddings tasks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn

from act.front_end.spec_creator_base import BaseSpecCreator, LabeledInputTensor
from act.front_end.specs import InKind, InputSpec, OutKind, OutputSpec
from act.front_end.bert_loader.data_loader import (
    BertEmbeddingClassifier,
    BertVocabulary,
    find_bert_dataset_name,
    load_bert_dataset,
    sample_correctly_classified,
)

logger = logging.getLogger(__name__)


class BertSpecCreator(BaseSpecCreator):
    """Create verification specifications from BERT dataset-model pairs."""

    def __init__(
        self,
        config_name: str | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the BERT specification creator.

        Args:
            config_name: Optional YAML config name.
            config_dict: Runtime configuration overrides.
        """
        super().__init__(config_name, config_dict)

    def create_specs_for_data_model_pairs(
        self,
        max_samples: int | None = None,
        filter_fn: Callable[[str, str], bool] | None = None,
        validate_shapes: bool = True,
        *,
        dataset_names: list[str] | None = None,
        model_names: list[str] | None = None,
        num_samples: int = 1,
        split: str = "test",
        max_verify_length: int = 8,
        epsilon: float = 0.1,
        p_norm: float = 1.0,
        perturbed_words: int = 1,
    ) -> list[tuple[str, str, nn.Module, list[torch.Tensor], list[tuple[InputSpec, OutputSpec]]]]:
        """Create embedding-space specs for BERT datasets.

        Args:
            dataset_names: Dataset names, defaulting to SST.
            model_names: Model names; only ``embedding_classifier`` is built in.
            num_samples: Number of correctly classified examples per pair.
            split: Dataset split to read.
            max_verify_length: Maximum token sequence length to verify.
            epsilon: Embedding-space perturbation radius.
            p_norm: Lp norm metadata carried by ``LP_EMBEDDING``.
            perturbed_words: Number of token positions to perturb from the start.
            validate_shapes: Whether to validate spec/model shape compatibility.

        Returns:
            List of ``(dataset, model_name, model_from_embeddings, labeled_embeddings, spec_pairs)``.
        """
        if max_samples is not None:
            num_samples = max_samples
        datasets = [find_bert_dataset_name(name) for name in (dataset_names or ["sst"])]
        models = model_names or ["embedding_classifier"]
        results: list[
            tuple[str, str, nn.Module, list[LabeledInputTensor], list[tuple[InputSpec, OutputSpec]]]
        ] = []

        for dataset in datasets:
            examples = load_bert_dataset(dataset, split)
            vocabulary = BertVocabulary(
                examples,
                embedding_dim=int(self.config.get("embedding_dim", 8)),
            )
            for model_name in models:
                if filter_fn is not None and not filter_fn(dataset, model_name):
                    continue
                if model_name != "embedding_classifier":
                    logger.warning("Skipping unsupported BERT model '%s'", model_name)
                    continue
                model = BertEmbeddingClassifier().eval()
                selected = sample_correctly_classified(
                    examples,
                    model,
                    vocabulary,
                    num_samples=num_samples,
                    max_verify_length=max_verify_length,
                )
                labeled_embeddings = [
                    LabeledInputTensor(
                        tensor=embeddings,
                        label=torch.tensor([predicted], dtype=torch.int64),
                    )
                    for _, embeddings, predicted in selected
                ]
                spec_pairs = [
                    self._create_spec_pair(
                        embeddings=embeddings,
                        predicted=predicted,
                        epsilon=epsilon,
                        p_norm=p_norm,
                        perturbed_words=perturbed_words,
                    )
                    for _, embeddings, predicted in selected
                ]
                if validate_shapes:
                    spec_pairs = self._validate_and_filter_specs(
                        spec_pairs, model, labeled_embeddings[0].tensor
                    )
                if spec_pairs:
                    results.append((dataset, model_name, model, labeled_embeddings, spec_pairs))

        return cast(
            list[tuple[str, str, nn.Module, list[torch.Tensor], list[tuple[InputSpec, OutputSpec]]]],
            results,
        )

    def _create_spec_pair(
        self,
        *,
        embeddings: torch.Tensor,
        predicted: int,
        epsilon: float,
        p_norm: float,
        perturbed_words: int,
    ) -> tuple[InputSpec, OutputSpec]:
        """Create one embedding input/output robustness spec pair."""
        length = embeddings.shape[-2]
        count = max(0, min(perturbed_words, length))
        positions = torch.arange(count, dtype=torch.long)
        input_spec = InputSpec(
            kind=InKind.LP_EMBEDDING,
            center=embeddings.clone(),
            eps=torch.tensor([epsilon], dtype=embeddings.dtype),
            p_norm=p_norm,
            perturbed_positions=positions,
        )
        output_spec = OutputSpec(
            kind=OutKind.MARGIN_ROBUST,
            y_true=torch.tensor([predicted], dtype=torch.int64),
            margin=torch.tensor([0.0], dtype=embeddings.dtype),
        )
        return input_spec, output_spec

    def _validate_and_filter_specs(
        self,
        spec_pairs: list[tuple[InputSpec, OutputSpec]],
        pytorch_model: nn.Module,
        sample_input: torch.Tensor,
    ) -> list[tuple[InputSpec, OutputSpec]]:
        """Validate BERT spec pairs against the embedding-space model."""
        valid_pairs = []
        for input_spec, output_spec in spec_pairs:
            is_valid, errors = self.validate_spec_pair_with_model(
                input_spec, output_spec, pytorch_model, sample_input
            )
            if is_valid:
                valid_pairs.append((input_spec, output_spec))
            else:
                logger.debug("BERT spec validation failed: %s", errors)
        return valid_pairs


def create_bert_specs(
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
    num_samples: int = 1,
) -> list[tuple[str, str, nn.Module, list[torch.Tensor], list[tuple[InputSpec, OutputSpec]]]]:
    """Create BERT specs with default creator settings.

    Args:
        dataset_names: BERT datasets to load.
        model_names: BERT model names to build.
        num_samples: Number of examples per dataset-model pair.

    Returns:
        BERT specification tuples in the unified front-end format.
    """
    creator = BertSpecCreator()
    return creator.create_specs_for_data_model_pairs(
        dataset_names=dataset_names,
        model_names=model_names,
        num_samples=num_samples,
    )
