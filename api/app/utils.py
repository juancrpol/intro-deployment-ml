import os
from io import BytesIO

from joblib import load
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


def get_model() -> Pipeline:
    model_path = os.environ.get("MODEL_PATH", "model/model.pkl")
    with open(model_path, "rb") as model_file:
        model = load(BytesIO(model_file.read()))
    return model


def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    transition_dictionary = {key: [value] for key, value in class_model.dict().items()}
    data_frame = DataFrame(transition_dictionary)
    return data_frame
