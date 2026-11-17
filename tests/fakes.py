"""In-memory stand-ins for MongoDB, and SYNTHETIC unit vectors used only to test
server-side logic. Real embeddings come from the browser model."""
import numpy as np

from backend.database import RepositoryError


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def make_person(seed, dim=128):
    return unit(np.random.default_rng(seed).normal(size=dim))


def samples_of(center, seed, n=3, noise=0.15):
    rng = np.random.default_rng(seed)
    return np.stack([unit(center + noise * unit(rng.normal(size=center.size))) for _ in range(n)])


class FakeRepo:
    def __init__(self):
        self.docs, self.down = [], False

    def _chk(self):
        if self.down:
            raise RepositoryError("simulated outage")

    def ping(self):
        self._chk()

    def count(self):
        self._chk()
        return len(self.docs)

    def load_embeddings(self, model_name, model_version):
        self._chk()
        return [{"id": str(i), "name": d["name"], "embeddings": np.asarray(d["embeddings"])}
                for i, d in enumerate(self.docs)
                if d["model_name"] == model_name and d["model_version"] == model_version]

    def insert(self, name, embeddings, model_name, model_version):
        self._chk()
        self.docs.append({"name": name, "embeddings": [list(map(float, e)) for e in embeddings],
                          "model_name": model_name, "model_version": model_version})
        return str(len(self.docs))
