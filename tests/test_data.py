from churn_prediction.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_dataset


def test_generated_dataset_has_expected_shape_and_binary_target():
    dataset = generate_dataset(rows=250, seed=7)

    assert list(dataset.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert len(dataset) == 250
    assert set(dataset[TARGET_COLUMN].unique()) == {0, 1}


def test_generation_is_reproducible():
    first = generate_dataset(rows=100, seed=19)
    second = generate_dataset(rows=100, seed=19)

    assert first.equals(second)
