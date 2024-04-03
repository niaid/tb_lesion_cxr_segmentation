import argparse
import json
from ultralytics import YOLO

def trainModel(pretrained_weights,yaml_file,model_info):
    
    model = YOLO(pretrained_weights)

    # Train the model
    model.train(data=yaml_file, epochs=model_info["epochs"], imgsz=model_info["imgsz"],name=model_info["name"],fliplr=model_info["fliplr"])
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_weights', type=str, help='Weights path')
    parser.add_argument('yaml_file', type=str, help='YAML file containing dataset path')
    parser.add_argument('model_info_json_path', type=str, help='Model info path')
    args = parser.parse_args()

    with open(str(args.model_info_json_path)) as f:
        model_info = json.load(f)
        
    trainModel(args.pretrained_weights,args.yaml_file,model_info)
   
if __name__ == "__main__":
    main()

