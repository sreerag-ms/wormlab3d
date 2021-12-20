import base64
from io import BytesIO

import numpy as np
from PIL import Image
from app.views.api import bp_api
from wormlab3d.data.model import Frame


@bp_api.route('/frame/<string:_id>', methods=['GET'])
def get_frame_data(_id):
    frame = Frame.objects.get(id=_id)

    # Read each numpy array into a PIL Image, write the png image to a buffer and encode as a base64-encoded string
    images = []
    for img_array in frame.images:
        z = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(z, 'L')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        data_uri = base64.b64encode(buffer.read()).decode('utf-8')
        images.append(f'data:image/png;charset=utf-8;base64,{data_uri}')

    response = {
        'images': images,
    }

    return response
