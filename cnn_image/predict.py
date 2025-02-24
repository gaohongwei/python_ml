# AIfDj8hEjGUIqEnHeagc
import logging
from typing import Sequence

import numpy as np
from sklearn.pipeline import Pipeline

from handlers.ml.config.defs import *
from handlers.ml.data_encoder.data_encoder import DataEncoder
from handlers.ml.pipeline_scorer import get_pipeline_scores

JSON_DATA = List[Dict[str, Any]]

logger = logging.getLogger(__name__)

import logging
import pickle
import torch

from handlers.ml.predict.predict_and_score import predict_and_score

logger = logging.getLogger(__name__)

def get_pipeline_scores(
    pipeline: Pipeline,
    x: Sequence,
    y: Sequence,
    opt_type: OptType,
    metrics_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    # y_enc: Optional[LabelEncoder] = None,
) -> Dict[str, Any]:
    """Evaluate the pipeline based on multiple metrics
    evaluate the metrics one by one and
    return in the format of dict[metric_name, metric_result]

    example usage:
    get_pipeline_scores(
        pipeline=some_classifier,
        x=X, y=y,
        metrics_kwargs={
            'roc_auc_score': {multi_class': 'ovo'}
        }
    )

    """

    if x is None or len(x) < 1:
        return {}

    y_pred = pipeline.predict(x)
    try:
        y_score = pipeline.predict_proba(x)
    except AttributeError:
        y_score = None

    # y_pred exist always, while y/y_true may not
    is_multiple_label = get_is_multiple_label_from_y_values(y_pred)
    all_eval_metrics = get_metrics(opt_type, is_multiple_label)

    return calculate_metric_scores(
        y_true=y,
        y_pred=y_pred,
        metrics=all_eval_metrics,
        y_score=y_score,
    )

def run_predict_tabular(config):
    model_encoder_path = config["model_data_path"]
    model_list = config["model_list"]
    file_path = config.get("file_path", None)
    json_data = config.get("json_data", None)
    logger.info(f"predict_data: {config}")

    try:
        with open(model_encoder_path, "rb") as data_encoder_fp:
            data_encoder = pickle.load(data_encoder_fp)
    except Exception as e:
        logger.error(f"Failed to load data encoder from {model_encoder_path}: {e}")
        return []  # or handle the error as needed

    predict_output = []
    # Load before loop to avoid loading 1+ times
    X, y_true = data_encoder.load_encode_data_for_predict(file_path, json_data)

    for one_model in model_list:
        model_id = one_model["model_id"]
        model_file_path = one_model["model_file_path"]

        try:
            # Load pipeline using a with statement
            with open(model_file_path, "rb") as model_fp:
                pipeline = pickle.load(model_fp)

            # Start prediction
            predict_details = predict_and_score(
                data=data_encoder, pipeline=pipeline, X=X, y_true=y_true
            )
            predict_output.append(
                {"model_id": model_id, "predict_details": predict_details}
            )

        except Exception as e:
            logger.debug(f"Skipping model ID {model_id} due to error: {e}")
            continue
    return predict_output

def predict_and_score(
    data: DataEncoder,
    pipeline: Pipeline,
    X: Sequence,
    y_true: Optional[Sequence] = None,
):
    # y_true has been encoded
    y_pred = pipeline.predict(X)
    y_pred_values = decode_predicted_values(y_pred, data=data)

    if y_true is None:
        return {"values": y_pred_values}

    # y_true has values
    exp_config = data._orig_config
    scores = get_pipeline_scores(
        pipeline=pipeline,
        x=X,
        y=y_true,
        opt_type=exp_config.opt_type,
    )

    # score and score_metric should be paired
    predict_details = {
        "scores": scores,
        "target_column_values": data.get_targe_column_values(),
        "score": scores.get(exp_config.opt_metric, None),
        "score_metric": exp_config.opt_metric,
        "values": y_pred_values,
    }

    # prepare y values for json support
    predict_details_json = convert_numpy_to_list(predict_details)
    return predict_details_json


def decode_predicted_values(
    y_pred,
    data: DataEncoder,
):
    logger.info(f"decode_predicted_values y_pred={y_pred}")
    logger.info(f"decode_predicted_values data={data}")
    if not data:
        logger.info(f"No DataEncoder")
        return to_list(y_pred)
    y_pred_decoded = data.decode_values(y_pred)
    logger.info(f"y_pred_decoded={y_pred_decoded}")
    y_pred_values = to_list(y_pred_decoded)
    logger.info(f"y_pred_values={y_pred_values}")
    return y_pred_values


def to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return data
    else:
        return [data]  # Handle the case where data is a single value


def convert_numpy_to_list(obj):
    """Recursively convert NumPy arrays to nested Python lists"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


def end_to_end_predict_old(
    data: DataEncoder,
    pipeline: Pipeline,
    file_path: Optional[str] = None,
    json_data: Optional[JSON_DATA] = None,
):
    """End-to-end prediction, from data path to predicted labels if possible"""
    _x, _ = data.load_encode_data(file_path, json_data)
    _y_pred = pipeline.predict(_x)
    if data.data_type == DataType.TAB and data.tab_y_col in data.column_encoders:
        return data.column_encoders[data.tab_y_col].inverse_transform(_y_pred)
    elif data.data_type == DataType.TXT and data.txt_y_col in data.column_encoders:
        return data.column_encoders[data.txt_y_col].inverse_transform(_y_pred)
    elif (
        data.data_type == DataType.IMG
        and data.img_class_enc_key in data.column_encoders
    ):
        return data.column_encoders[data.img_class_enc_key].inverse_transform(_y_pred)
    else:
        return _y_pred


def predict_with_details_old(
    data: DataEncoder,
    pipeline: Pipeline,
    path: str,
    file_path: Optional[str] = None,
    json_data: Optional[JSON_DATA] = None,
):
    """
    Old code, Keep it for reviewing
    """
    _x, _ = data.load(file_path, json_data)
    _y_pred = end_to_end_predict_old(data, pipeline, file_path, json_data)

    # dictionary that maps pipeline attribute to the name in the detail dictionary
    _detail_attr = {"predict_proba": "proba", "predict_log_proba": "log_proba"}

    _detail_dict = {}
    for __attr, __detail_name in _detail_attr.items():
        try:
            _detail_dict[__detail_name] = pipeline.__getattribute__(__attr)(_x)
        except AttributeError:
            _detail_dict[__detail_name] = None
    return _y_pred, _detail_dict