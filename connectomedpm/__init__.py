"""ConnectomeDPM: connectome-prior selective parameter correction.

The package is deliberately split so that every experimental claim maps to a module:

    connectomedpm.data          task registry, dedup, frozen splits
    connectomedpm.backbone      frozen base model, feature extraction, generation
    connectomedpm.blocks        low-rank parameter-correction blocks and injection
    connectomedpm.graph_prior   MaleCNS loader, coarse-graining, operators, null models
    connectomedpm.supervision   scorer, FIX/BREAK/UNCHANGED labels, candidate cache
    connectomedpm.router        fixed transfer operator + action head
    connectomedpm.calibration   deployment threshold, risk and coverage estimation
    connectomedpm.evaluation    fixed-coverage metrics, clustered bootstrap
    connectomedpm.audit         routing traces, graph-usage and reproducibility audits
"""

__version__ = "0.1.0"

