import base64
import datetime
import json
from io import BytesIO

import numpy as np
from PIL import Image
from bson import ObjectId


class JSONEncoder(json.JSONEncoder):
    """
    Extends the default json encoder to encode datetime objects and mongodb (bson) object ids as strings.
    """

    def default(self, z):
        if isinstance(z, datetime.datetime):
            return str(z)
        elif isinstance(z, ObjectId):
            return str(z)
        else:
            return super().default(z)


def base64img(z: np.ndarray, image_mode: str = 'RGBA') -> str:
    """
    Read image array into a PIL Image, write the png image to a buffer and encode as a base64-encoded string.
    """
    img = Image.fromarray(z, image_mode)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    data_uri = base64.b64encode(buffer.read()).decode('utf-8')
    img_str = f'data:image/png;charset=utf-8;base64,{data_uri}'
    return img_str
