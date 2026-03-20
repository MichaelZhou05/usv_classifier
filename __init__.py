"""USV Classifier - Disease detection from mouse ultrasonic vocalizations.

Supports two modes:
1. Legacy mode: MAT files with basic features (binary classification)
2. Enriched mode: CSV files with pooled features (3-class classification)

Quick Start (Enriched Mode):
    # 1. Export features from MATLAB
    ExportEnrichedFeaturesDir('./DeepSqueak_output', './enriched_features')

    # 2. Train classifier
    python train.py --config config_enriched.yaml

    # Or programmatically:
    from usv_classifier.pooling import PoolerRegistry
    from usv_classifier.data import EnrichedUSVDataset
    from usv_classifier.models import EnrichedUSVClassifier

    pooler = PoolerRegistry.get("average", n_features=11)
    dataset = EnrichedUSVDataset("./enriched_features", pooler=pooler)
    model = EnrichedUSVClassifier(input_dim=11, n_classes=3)
"""

__version__ = "0.2.0"
