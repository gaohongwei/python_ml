"""The network itself, and the artifact that makes it reusable.

Everything a caller outside this package needs is re-exported here, so the inner
module names stay an implementation detail:

    from tcn_model import TcnModel, build_tcn_model      # the network
    from tcn_model import save_model_artifact, load_model_artifact, TLoadedModel

Inside the package, keep importing the modules directly
(`from tcn_model.tcn_model import TcnModel`) - going through this file would make
the package import itself while it is still being defined.

Not exported: TrimRightPadding, TemporalBlock, build_causal_conv. They are parts
of the network, not things to build with.
"""

from tcn_model.tcn_model import TcnModel, build_tcn_model, count_parameters
from tcn_model.artifact import (
    ARTIFACT_FORMAT_VERSION,
    MODEL_TYPE,
    TLoadedModel,
    build_preprocess_dict,
    load_model_artifact,
    save_model_artifact,
)

__all__ = [
    # network
    "TcnModel",
    "build_tcn_model",
    "count_parameters",
    # artifact
    "TLoadedModel",
    "save_model_artifact",
    "load_model_artifact",
    "build_preprocess_dict",
    "ARTIFACT_FORMAT_VERSION",
    "MODEL_TYPE",
]
