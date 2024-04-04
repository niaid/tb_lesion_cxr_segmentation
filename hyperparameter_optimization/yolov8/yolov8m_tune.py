from ultralytics import YOLO
import argparse
import json


def tune_model(pretrained_weights, yaml_file, model_info):
    # Initialize the YOLO model
    model = YOLO("yolov8m.pt")

    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(
        data="/data/bcbb/kantipudik2/lesion_segmentation/zhying/yolov8_dataset/lesion_segment.yaml",
        epochs=model_info["epochs"],
        iterations=model_info["iterations"],
        plots=True,
        save=True,
        val=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_yaml_file", type=str, help="Input yaml file")
    parser.add_argument("model_info_json_path", type=str, help="Model info")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to save\
                                                      the images and labels \
                                                      corresponding to \
                                                      yolov8",
    )
    args = parser.parse_args()

    with open(str(args.model_info_json_path)) as f:
        model_info = json.load(f)

    tune_model(args.pretrained_weights, args.yaml_file, model_info)


if __name__ == "__main__":
    main()
