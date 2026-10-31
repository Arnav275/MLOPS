import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml.config import load_params, path  # noqa: E402


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def modelling(params):
    p = path(params["data"]["modelling_dataset"])
    if not p.exists():
        from ml.data import prepare
        return prepare(params)
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def bundle(params):
    from ml.bundle import load_bundle
    d = path(params["serving"]["champion_dir"])
    if not (d / "model.joblib").exists():
        pytest.skip("no champion bundle - run `python -m ml.train` first")
    return load_bundle(d)


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient

    from api.main import create_app
    with TestClient(create_app(db_path=tmp_path / "mon.db")) as c:
        yield c


@pytest.fixture()
def sample_record(modelling):
    r = modelling[modelling.split == "test"].iloc[0].drop(["group_id", "split", "fail"]).to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
