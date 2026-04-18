"""Unit tests for data splits module."""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realwaste_mlops.data.splits import (
    create_split_manifests,
    write_split_manifest,
    write_all_split_manifests,
)
from realwaste_mlops.data.validate import DatasetRecord


class TestCreateSplitManifests:
    """Test create_split_manifests function."""

    def test_split_proportions(self):
        """Test that splits have correct proportions."""
        # Create sample data - 10 records per class
        records = []
        for class_name in ["Metal", "Cardboard"]:
            for i in range(10):
                records.append(
                    DatasetRecord(
                        file_path=Path(f"/path/to/{class_name}_{i}.jpg"),
                        class_name=class_name,
                    )
                )

        split_map = create_split_manifests(
            records=records,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        # Check counts - 10 per class, so 7 train, 1 val, 2 test per class
        assert len(split_map["train"]) == 14  # 7 * 2 classes
        assert len(split_map["val"]) == 2  # 1 * 2 classes
        assert len(split_map["test"]) == 4  # 2 * 2 classes

    def test_stratification(self):
        """Test that stratification is maintained."""
        records = []
        for class_name in ["Metal", "Cardboard"]:
            for i in range(10):
                records.append(
                    DatasetRecord(
                        file_path=Path(f"/path/to/{class_name}_{i}.jpg"),
                        class_name=class_name,
                    )
                )

        split_map = create_split_manifests(
            records=records,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        # Each class should be represented in each split
        train_classes = set(r.class_name for r in split_map["train"])

        assert train_classes == {"Metal", "Cardboard"}

    def test_empty_input(self):
        """Test with empty input."""
        split_map = create_split_manifests(
            records=[],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        assert len(split_map["train"]) == 0
        assert len(split_map["val"]) == 0
        assert len(split_map["test"]) == 0

    def test_single_class(self):
        """Test with single class."""
        records = [
            DatasetRecord(file_path=Path(f"/path/to/img_{i}.jpg"), class_name="Metal")
            for i in range(10)
        ]

        split_map = create_split_manifests(
            records=records,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        # Should have 7 train, 1 val, 2 test
        assert len(split_map["train"]) == 7
        assert len(split_map["val"]) == 1
        assert len(split_map["test"]) == 2


class TestWriteSplitManifest:
    """Test write_split_manifest function."""

    def test_write_manifest(self, tmp_path):
        """Test writing a manifest file."""
        records = [
            DatasetRecord(file_path=Path("/path/to/img.jpg"), class_name="Metal")
        ]

        output_path = tmp_path / "test.json"
        write_split_manifest("train", records, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["split"] == "train"
        assert data["count"] == 1
        assert len(data["records"]) == 1
        assert data["records"][0]["class_name"] == "Metal"


class TestWriteAllSplitManifests:
    """Test write_all_split_manifests function."""

    def test_write_all_manifests(self, tmp_path):
        """Test writing all split manifests."""
        records = [
            DatasetRecord(file_path=Path(f"/path/to/img_{i}.jpg"), class_name="Metal")
            for i in range(10)
        ]

        split_map = create_split_manifests(
            records=records,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        manifests_dir = tmp_path / "manifests"
        write_all_split_manifests(split_map, manifests_dir)

        # Check files exist
        assert (manifests_dir / "train.json").exists()
        assert (manifests_dir / "val.json").exists()
        assert (manifests_dir / "test.json").exists()
