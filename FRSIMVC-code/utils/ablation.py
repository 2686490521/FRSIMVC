"""Controllable ablation switches for the FSRIMVC pipeline.

Outline-v3 ablation ladder.  Every rung is evaluated on the *same* pre-SLIC
corrupted input, so A0 is now the true "Original FedRSMVC under corrupted
remote-sensing observations" baseline:

    A0 = Original FedRSMVC + pre-SLIC missing   (naive; no mask awareness)
    A1 = A0 + mask-aware reconstruction / KNN   (observed-only local graph)
    A2 = A1 + consensus spatial graph           (weight graph_beta)
    A3 = A2 + CSATA spatial signature + Sinkhorn
    A4 = A3 + global semantic consensus         (RESERVED / closed by default)

A4 is kept runnable for diagnostics but is closed in the main protocol: the
global-consensus term contributed the least in the 1-epoch scheduling check
and it is the component currently under questioning.

Every experiment is driven by four independent boolean switches.  Passing
``--ablation-variant`` sets them together; without it the config's own
``mask_aware`` / ``spatial_graph`` / ``csata_alignment`` / ``global_consensus``
keys are used (all default to True, i.e. the unchanged full model behaviour).
"""

import warnings

FLAG_KEYS = ('mask_aware', 'spatial_graph', 'csata_alignment', 'global_consensus')

VARIANT_FLAGS = {
    'A0': {'mask_aware': False, 'spatial_graph': False, 'csata_alignment': False, 'global_consensus': False},
    'A1': {'mask_aware': True,  'spatial_graph': False, 'csata_alignment': False, 'global_consensus': False},
    'A2': {'mask_aware': True,  'spatial_graph': True,  'csata_alignment': False, 'global_consensus': False},
    'A3': {'mask_aware': True,  'spatial_graph': True,  'csata_alignment': True,  'global_consensus': False},
    'A4': {'mask_aware': True,  'spatial_graph': True,  'csata_alignment': True,  'global_consensus': True},
}

VARIANT_DESCRIPTION = {
    'A0': 'Original FedRSMVC + pre-SLIC missing (naive, no mask awareness)',
    'A1': 'A0 + mask-aware reconstruction / observed-only local KNN graph',
    'A2': 'A1 + consensus spatial graph fusion (graph_beta)',
    'A3': 'A2 + CSATA spatial signature + Sinkhorn alignment',
    'A4': 'A3 + global semantic consensus (RESERVED: closed in the main protocol)',
}

# Rungs that stay runnable but are not part of the main experiment matrix.
CLOSED_VARIANTS = ('A4',)


def _clean_variant(value):
    if value is None:
        return None
    name = str(value).strip()
    if name == '' or name.lower() in ('none', 'null', 'false'):
        return None
    return name.upper()


def resolve_ablation(config):
    """Return ``(variant, flags)``.

    ``variant`` is the canonical name (e.g. ``'A2'``) or ``None`` for the full
    model.  ``flags`` always contains the four switch booleans plus
    ``mask_aware_loss`` (defaults to ``mask_aware``).
    """
    variant = _clean_variant(config.get('ablation_variant'))
    if variant is not None:
        if variant not in VARIANT_FLAGS:
            raise ValueError(
                f'Unknown ablation_variant {config.get("ablation_variant")!r}; '
                f'expected one of {sorted(VARIANT_FLAGS)}'
            )
        # The rung preset is authoritative: A0 must stay fully naive even if
        # the config still carries explicit mask-aware keys.
        flags = dict(VARIANT_FLAGS[variant])
        flags['mask_aware_loss'] = flags['mask_aware']
        flags['mask_aware_eval'] = flags['mask_aware']
        if variant in CLOSED_VARIANTS:
            warnings.warn(
                f'{variant} is closed in the main protocol '
                f'({VARIANT_DESCRIPTION[variant]}); running it for diagnostics only.',
                stacklevel=2,
            )
    else:
        flags = {key: bool(config.get(key, True)) for key in FLAG_KEYS}
        # Without a rung, the two mask-aware sub-switches can be tuned
        # independently for finer-grained diagnostics.
        flags['mask_aware_loss'] = bool(config.get('mask_aware_loss', flags['mask_aware']))
        flags['mask_aware_eval'] = bool(config.get('mask_aware_eval', flags['mask_aware']))
    return variant, flags


def variant_label(variant, flags):
    """Human readable one-liner used in logs and summaries."""
    head = variant if variant else 'FULL(V5)'
    body = ' '.join(f"{key.split('_')[0]}={'on' if flags[key] else 'off'}" for key in FLAG_KEYS)
    return f'{head} [{body}]'
