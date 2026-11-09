"""Compatibility helpers for loading pickled scikit-learn models."""

import sys

from loguru import logger


def register_legacy_loss_module() -> bool:
    """Registers scikit-learn's Cython loss module under its pre-1.9 name.

    Models pickled by scikit-learn 1.8 and earlier, such as the SGD classifier,
    store their loss function as ``_loss.CyHalfBinomialLoss``. Those releases
    registered the top-level module name ``_loss``. Release 1.9 no longer does,
    so ``joblib.load`` fails with ``No module named '_loss'``. Registering the
    alias before loading lets the existing pickles load without retraining.

    Returns:
        True if the alias is registered after the call, False if scikit-learn
        does not provide the loss module.
    """
    if "_loss" in sys.modules:
        return True
    try:
        from sklearn._loss import _loss
    except ImportError:
        logger.warning("sklearn._loss._loss is unavailable; SGD models may fail to load")
        return False
    sys.modules["_loss"] = _loss
    return True
