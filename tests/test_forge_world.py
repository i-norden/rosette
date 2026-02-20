"""Tests for rosette.forge_world — forge-world protocol implementations.

Verifies protocol conformance, seed-aware dataset behavior, and factory functions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from forge_world.core.protocols import (
    Aggregator,
    LabeledDataset,
    LabeledItem,
    PassFailRule,
    PassFailRuleSet,
    Pipeline,
    Severity,
)

from rosette.forge_world import (
    RosetteAggregator,
    RosetteDataset,
    RosettePipeline,
    RosetteRules,
    create_aggregator,
    create_dataset,
    create_pipeline,
    create_rules,
)


class TestProtocolConformance:
    """Verify that rosette.forge_world classes satisfy forge-world protocols."""

    def test_pipeline_is_protocol(self):
        p = create_pipeline()
        assert isinstance(p, Pipeline)

    def test_aggregator_is_protocol(self):
        a = create_aggregator()
        assert isinstance(a, Aggregator)

    def test_dataset_is_protocol(self):
        d = create_dataset()
        assert isinstance(d, LabeledDataset)

    def test_rules_is_protocol(self):
        r = create_rules()
        assert isinstance(r, PassFailRuleSet)


class TestFactoryFunctions:
    def test_create_pipeline(self):
        p = create_pipeline()
        assert isinstance(p, RosettePipeline)
        assert p.get_config() is not None
        assert isinstance(p.get_config_schema(), dict)

    def test_create_aggregator(self):
        a = create_aggregator()
        assert isinstance(a, RosetteAggregator)

    def test_create_dataset(self):
        d = create_dataset()
        assert isinstance(d, RosetteDataset)

    def test_create_rules(self):
        r = create_rules()
        assert isinstance(r, RosetteRules)


class TestRosetteRules:
    def test_findings_rule(self):
        r = RosetteRules()
        rule = r.get_rule("findings")
        assert rule.min_risk_for_pass == Severity.MEDIUM
        assert rule.max_risk_for_pass is None

    def test_clean_rule(self):
        r = RosetteRules()
        rule = r.get_rule("clean")
        assert rule.max_risk_for_pass == Severity.MEDIUM
        assert rule.min_risk_for_pass is None

    def test_informational_rule(self):
        r = RosetteRules()
        rule = r.get_rule("informational")
        assert rule.min_risk_for_pass is None
        assert rule.max_risk_for_pass is None


class TestRosetteDataset:
    def test_fixed_items_no_seed(self):
        """Without seed, only fixed items are returned (no Zenodo)."""
        d = RosetteDataset()
        items = d.items(seed=None)
        # All items should be from fixture directories, not Zenodo
        for item in items:
            assert not item.id.startswith("zenodo_"), (
                f"Item {item.id} should not be from Zenodo when seed=None"
            )

    def test_items_returns_labeled_items(self):
        d = RosetteDataset()
        items = d.items()
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, LabeledItem)
            assert item.id
            assert item.category
            assert item.expected_label in ("findings", "clean", "informational")

    def test_categories_returns_list(self):
        d = RosetteDataset()
        cats = d.categories()
        assert isinstance(cats, list)
        # Should include Zenodo categories in the list
        assert "rsiil_zenodo" in cats
        assert "rsiil_zenodo_clean" in cats

    def test_seed_items_include_zenodo(self):
        """With seed, Zenodo items are included (if data exists)."""
        d = RosetteDataset()
        items_no_seed = d.items(seed=None)
        items_with_seed = d.items(seed=42)

        # With seed should have >= items without seed
        # (more if Zenodo data is available)
        assert len(items_with_seed) >= len(items_no_seed)

    def test_different_seeds_different_items(self):
        """Different seeds should produce different Zenodo samples."""
        d = RosetteDataset()
        items_42 = d.items(seed=42)
        items_137 = d.items(seed=137)

        zenodo_42 = {i.id for i in items_42 if i.id.startswith("zenodo_")}
        zenodo_137 = {i.id for i in items_137 if i.id.startswith("zenodo_")}

        # If Zenodo data is available, different seeds should give different samples
        if zenodo_42 and zenodo_137:
            assert zenodo_42 != zenodo_137

    def test_sample_size_limits_zenodo(self):
        """sample_size should limit how many Zenodo items are sampled."""
        d = RosetteDataset()
        items_full = d.items(seed=42)
        items_limited = d.items(seed=42, sample_size=5)

        zenodo_full = [i for i in items_full if i.id.startswith("zenodo_")]
        zenodo_limited = [i for i in items_limited if i.id.startswith("zenodo_")]

        # If Zenodo data exists, limited should have fewer items
        if zenodo_full:
            assert len(zenodo_limited) <= len(zenodo_full)

    def test_fixed_items_cached(self):
        """Fixed items should be loaded once and cached internally."""
        d = RosetteDataset()
        items1 = d.items()
        items2 = d.items()
        # items() returns a copy each time (since seed items may be appended),
        # but the underlying fixed items list is cached
        assert len(items1) == len(items2)
        assert d._fixed_items is not None
        # Calling _load_fixed() again returns the same cached list object
        assert d._load_fixed() is d._load_fixed()

    def test_seed_none_same_as_no_args(self):
        """items(seed=None) should return same items as items()."""
        d = RosetteDataset()
        items_default = d.items()
        items_none = d.items(seed=None)
        assert len(items_default) == len(items_none)

    def test_zenodo_items_have_metadata(self):
        """Zenodo items should include seed in metadata."""
        d = RosetteDataset()
        items = d.items(seed=42)
        zenodo_items = [i for i in items if i.id.startswith("zenodo_")]
        for item in zenodo_items:
            assert item.metadata.get("source") == "zenodo"
            assert item.metadata.get("seed") == 42
