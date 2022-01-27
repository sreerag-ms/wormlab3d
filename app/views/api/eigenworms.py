import numpy as np

from app.views.api import bp_api
from wormlab3d.data.model import Eigenworms


@bp_api.route('/eigenworms/<string:_id>', methods=['GET'])
def get_eigenworms(_id: str):
    eigenworms = Eigenworms.objects.get(id=_id)

    response = {
        'idxs': np.arange(eigenworms.n_components).tolist(),
        'singular_values': eigenworms.singular_values,
        'explained_variance_ratio': np.cumsum(eigenworms.explained_variance_ratio).tolist(),
    }

    return response
