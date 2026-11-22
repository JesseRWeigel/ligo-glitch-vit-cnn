"""Write a portable, row-aligned paired prediction archive for future O4 runs."""

import numpy as np


def save_o4_predictions(
    path, *, gravityspy_ids, expected_labels, cnn_labels, vit_labels,
    cnn_preds, vit_preds, class_names,
):
    """Validate dataset order and both model outputs before touching the archive."""
    ids = np.asarray(gravityspy_ids, dtype=object)
    if ids.ndim != 1 or not ids.size:
        raise ValueError("expected a nonempty, one-dimensional gravityspy_id list")
    if not all(isinstance(value, str) and value.strip() for value in ids):
        raise ValueError("each gravityspy_id must be a nonblank string")
    if len({value.strip() for value in ids}) != ids.size:
        raise ValueError("duplicate gravityspy_id in O4 evaluation")

    arrays = {}
    for name, values in (
        ("expected_labels", expected_labels), ("cnn_labels", cnn_labels),
        ("vit_labels", vit_labels), ("cnn_preds", cnn_preds), ("vit_preds", vit_preds),
    ):
        vector = np.asarray(values)
        if vector.shape != ids.shape or not np.issubdtype(vector.dtype, np.integer):
            raise ValueError(f"{name} needs one integer per gravityspy_id")
        if np.any(vector < 0) or np.any(vector >= len(class_names)):
            raise ValueError(f"{name} contains an unknown class index")
        arrays[name] = vector
    for name in ("cnn_labels", "vit_labels"):
        if not np.array_equal(arrays[name], arrays["expected_labels"]):
            raise ValueError(f"{name} does not match ordered O4 manifest labels")

    np.savez(
        path, y_true=arrays["expected_labels"], cnn_preds=arrays["cnn_preds"],
        vit_preds=arrays["vit_preds"], class_names=np.asarray(class_names, dtype=str),
        gravityspy_ids=np.asarray(ids, dtype=str),
    )
