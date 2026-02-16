# image_cache.py
import os, hashlib, requests
from io import BytesIO
from PIL import Image

class ImageCache:
    def __init__(self, cache_dir, timeout=5):
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, url):
        h = hashlib.sha1(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.jpg")

    def get(self, url):
        path = self._path(url)

        # Cache hit
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                os.remove(path)

        # Cache miss → download
        try:
            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(path, format="JPEG", quality=95)
            return img
        except Exception:
            return None
