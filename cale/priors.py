from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_priors(frequency_file, prior_file, num_classes, background):
    """Load the ordered foreground statistics used by the frozen classifier."""
    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p
    frequency = pd.read_csv(resolve(frequency_file))
    prior = pd.read_csv(resolve(prior_file))
    if len(frequency) != num_classes + 1 or len(prior) != num_classes:
        raise ValueError('Prior/category count does not match the classifier.')
    if (frequency['category_id'].tolist() != list(range(num_classes + 1)) or
            prior['category_id'].tolist() != list(range(1, num_classes + 1))):
        raise ValueError('Prior rows must follow the classifier category order.')
    # Row zero is the original background placeholder. base10 = -log10(f_c).
    target = np.ones(num_classes) * 300
    bias = frequency['base10'].values[1:] - np.log10(target.sum() / target)
    values = prior['lambda'].values.tolist()
    if not np.isfinite(bias).all() or not np.isfinite(values).all():
        raise ValueError('Non-finite prior entries.')
    if not np.all((np.asarray(values) >= 1) & (np.asarray(values) <= 2)):
        raise ValueError('Structural factors must lie in [1, 2].')
    bias = bias.tolist()
    if background:
        bias += [0.]
        values += [1.]
    return (torch.tensor(bias, dtype=torch.float32).unsqueeze(0),
            torch.tensor(values, dtype=torch.float32))
