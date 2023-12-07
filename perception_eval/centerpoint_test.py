import argparse
import logging
import os
import tempfile
from typing import List

from pyquaternion.quaternion import Quaternion

from perception_eval.common.label import Label
from perception_eval.common.label import LabelConverter
from perception_eval.common.label import LabelType
from perception_eval.common.object import DynamicObject
from perception_eval.common.shape import Shape
from perception_eval.common.shape import ShapeType
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.manager import PerceptionEvaluationManager
from perception_eval.perception_eval.common.schema import FrameID
from perception_eval.tool import PerceptionAnalyzer3D
from perception_eval.util.debug import format_class_for_log
from perception_eval.util.debug import get_objects_with_difference
from perception_eval.util.logger_config import configure_logger


class PerceptionLSimMoc:
    def __init__(
        self,
        dataset_paths: List[str],
        evaluation_task: str,
        result_root_directory: str,
    ):
        evaluation_config_dict = {
            "evaluation_task": evaluation_task,
            # ラベル，max x/y，マッチング閾値 (detection/tracking/predictionで共通)
            "target_labels": ["car", "truck", "bus", "bicycle", "pedestrian"],
            "ignore_attributes": ["cycle_state.without_rider"],
            # max x/y position or max/min distanceの指定が必要
            # # max x/y position
            "max_x_position": 102.4,
            "max_y_position": 102.4,
            # max/min distance
            # "max_distance": 102.4,
            # "min_distance": 10.0,
            # # confidenceによるフィルタ (Optional)
            # "confidence_threshold": 0.5,
            # # GTのuuidによるフィルタ (Optional)
            # "target_uuids": ["foo", "bar"],
            # objectごとにparamを設定
            "center_distance_thresholds": [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
            ],  # = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
            # objectごとに同じparamの場合はこのような指定が可能
            "plane_distance_thresholds": [
                2.0,
                3.0,
            ],  # = [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
            "iou_2d_thresholds": [0.5, 0.5, 0.5, 0.5, 0.5],  # = [[0.5, 0.5, 0.5, 0.5]]
            "iou_3d_thresholds": [0.5],  # = [[0.5, 0.5, 0.5, 0.5]]
            "min_point_numbers": [0, 0, 0, 0, 0],
            "max_matchable_radii": 5.0,  # = [5.0, 5.0, 5.0, 5.0]
            # label parameters
            "label_prefix": "autoware",
            "merge_similar_labels": True,
            "allow_matching_unknown": True,
        }

        evaluation_config = PerceptionEvaluationConfig(
            dataset_paths=dataset_paths,
            frame_id="base_link" if evaluation_task == "detection" else "map",
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
            load_raw_data=True,
        )

        _ = configure_logger(
            log_file_directory=evaluation_config.log_directory,
            console_log_level=logging.INFO,
            file_log_level=logging.INFO,
        )

        self.evaluator = PerceptionEvaluationManager(evaluation_config=evaluation_config)

    def callback(
        self,
        unix_time: int,
        estimated_objects: List[DynamicObject],
    ) -> None:
        # 現frameに対応するGround truthを取得
        ground_truth_now_frame = self.evaluator.get_ground_truth_now_frame(unix_time)

        # [Option] ROS側でやる（Map情報・Planning結果を用いる）UC評価objectを選別
        # ros_critical_ground_truth_objects : List[DynamicObject] = custom_critical_object_filter(
        #   ground_truth_now_frame.objects
        # )
        ros_critical_ground_truth_objects = ground_truth_now_frame.objects

        # 1 frameの評価
        # 距離などでUC評価objectを選別するためのインターフェイス（PerceptionEvaluationManager初期化時にConfigを設定せず、関数受け渡しにすることで動的に変更可能なInterface）
        # どれを注目物体とするかのparam
        critical_object_filter_config: CriticalObjectFilterConfig = CriticalObjectFilterConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "truck", "bus", "bicycle", "pedestrian"],
            ignore_attributes=["cycle_state.without_rider"],
            max_x_position_list=[30.0, 30.0, 30.0, 30.0, 30.0],
            max_y_position_list=[30.0, 30.0, 30.0, 30.0, 30.0],
        )
        # Pass fail を決めるパラメータ
        frame_pass_fail_config: PerceptionPassFailConfig = PerceptionPassFailConfig(
            evaluator_config=self.evaluator.evaluator_config,
            target_labels=["car", "truck", "bus", "bicycle", "pedestrian"],
            matching_threshold_list=[2.0, 2.0, 2.0, 2.0, 2.0],
        )

        frame_result = self.evaluator.add_frame_result(
            unix_time=unix_time,
            ground_truth_now_frame=ground_truth_now_frame,
            estimated_objects=estimated_objects,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
        )
        self.visualize(frame_result)

    def get_final_result(self) -> MetricsScore:
        """
        処理の最後に評価結果を出す
        """

        # number of fails for critical objects
        num_critical_fail: int = sum(
            map(
                lambda frame_result: frame_result.pass_fail_result.get_num_fail(),
                self.evaluator.frame_results,
            )
        )
        logging.info(f"Number of fails for critical objects: {num_critical_fail}")

        # scene metrics score
        final_metric_score = self.evaluator.get_scene_result()
        logging.info(f"final metrics result {final_metric_score}")
        return final_metric_score

    def visualize(self, frame_result: PerceptionFrameResult):
        """
        Frameごとの可視化
        """
        logging.info(
            f"{len(frame_result.pass_fail_result.tp_object_results)} TP objects, "
            f"{len(frame_result.pass_fail_result.fp_object_results)} FP objects, "
            f"{len(frame_result.pass_fail_result.fn_objects)} FN objects",
        )

        if frame_result.metrics_score.maps[0].map < 0.7:
            logging.debug("mAP is low")
            # logging.debug(f"frame result {format_class_for_log(frame_result.metrics_score)}")

        # Visualize the latest frame result
        # self.evaluator.visualize_frame()

    def from_detection(self, file_path, file_name, classes) -> List[DynamicObject]:
        detections = []
        file = open(file_path)
        for line in file:
            line = map(float, line.split())
            label = int(line[0])
            score = line[1]
            x, y, z = line[2], line[3], line[4]
            length, width, height = line[5], line[6], line[7]
            yaw = line[8]
            vel_x, vel_y = line[9], line[10]

            unix_time = 0
            frame_id = FrameID().from_value("base_link")
            for frame in self.evaluator.ground_truth_frames:
                if frame.frame_name == file_name:
                    unix_time = frame.unix_time
                    break

            frame_id = "base_link"
            orientation = Quaternion(axis=(0.0, 0.0, 1.0), radians=yaw)
            shape = Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(length, width, height))
            velocity = tuple(vel_x, vel_y, 0.0)
            position = tuple(x, y, z)
            uuid = file_name
            lc = LabelConverter("detection", False, "autoware")
            semantic_label = lc.convert_label(classes[label])
            detections.append(DynamicObject(unix_time, frame_id, position, orientation, shape, velocity, score, semantic_label, uuid=uuid))
        return detections


# REQUIRED:
#   dataset_path: str
#   model: Your 3D ML model

# evaluation_config = PerceptionEvaluationConfig(
#     dataset_paths=[dataset_path],
#     frame_id="base_link",
#     result_root_directory="./data/result",
#     evaluation_config_dict={"evaluation_task": "detection"},
#     load_raw_data=True,
# )

# # initialize Evaluation Manager
# evaluator = PerceptionEvaluationManager(evaluation_config=evaluation_config)

# critical_object_filter_config = CriticalObjectFilterConfig(...)
# pass_fail_config = PerceptionPassFailConfig(...)

# for frame in datasets:
#     unix_time = frame.unix_time
#     pointcloud: numpy.ndarray = frame.raw_data["lidar"]
#     outputs = model(pointcloud)
#     # create a list of estimated objects with your model's outputs
#     estimated_objects = [DynamicObject(unix_time=unix_time, ...) for out in outputs]
#     # add frame result
#     evaluator.add_frame_result(
#         unix_time=unix_time,
#         ground_truth_now_frame=frame,
#         estimated_objects=estimated_objects,
#         ros_critical_ground_truth_objects=frame.objects,
#         critical_object_filter_config=critical_object_filter_config,
#         frame_pass_fail_config=pass_fail_config,
#     )

# scene_score = evaluator.get_scene_result()


dataset_path = "/home/xinyuwang/adehome/eval_dataset/DBv2.0_nishi_shinjuku_6-3-9d847f22-a8e4-430c-918e-102341f01311/data"
result_root_directory = "/home/xinyuwang/adehome/eval_dataset/result"
detection_path = "/home/develop/dataset/detection/"
classes = ["car", "truck", "bus", "bicycle", "pedestrain"]
detection_lsim = PerceptionLSimMoc([dataset_path], "detection", result_root_directory)
for frame in detection_lsim.evaluator.ground_truth_frames:
    print(frame.frame_name)

for f in os.listdir(detection_path):
    print(f)
    detections = detection_lsim.from_detection(detection_path+f, f.split('.')[0], classes)
    if(len(detections) > 0):
        detection_lsim.callback(detections[0].unix_time, detections)
print(detection_lsim.get_final_result())
