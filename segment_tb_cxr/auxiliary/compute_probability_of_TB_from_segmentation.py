import numpy as np
import pandas as pd
import argparse
import sys
import SimpleITK as sitk

"""
This script computes the probabilities of TB for the enitre level image 
given the pixel probabilities of the image.  In some cases, YOLOv8 does not generate
any detected masks, and hence the probabilities of the image for only images where the model
was able to predict were computed. Rest all images where the model was not able to
predict any mask, we assume the probability of TB for that image to be zero.

"""

def compute_stats(probability_map):
        
    return np.min(probability_map),np.mean(probability_map), np.max(probability_map)

def get_prob_of_tb(tb_pred_arr_npz_file,threshold=0.5):
    
    # For all the models. 
    probability_map = sitk.GetArrayFromImage(sitk.ReadImage(tb_pred_arr_npz_file))
    filtered_tb_probabilities = probability_map[probability_map > threshold]
    
    if len(filtered_tb_probabilities) == 0:
        return compute_stats(probability_map)
    
    else:
        return compute_stats(filtered_tb_probabilities)

    
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv_path",
        help='Enter the csv files containing filenames with column names (processed_Filename,pred_tb_probability_arr_file_path,pred_lung_probability_arr_file_path')
    parser.add_argument(
        "--threshold",
        default=0.5,
        help='Default threshold for pixel to classify as TB  pixel")',
    )
    parser.add_argument("output_csv_file_path", help="Output CSV file path containing tb probabilities.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv_path)

    df[['probabilty_of_tb_from_min_overall',
        'probabilty_of_tb_from_mean_overall', 
        'probabilty_of_tb_from_max_overall']] = pd.DataFrame(
        df['nnunet_yolov8_cropped_pred_img_filenames'].apply(lambda x: get_prob_of_tb(x)).tolist(),
        index=df.index
    )
                            
    df.to_csv(args.output_csv_file_path,index=False)
    
if __name__ == "__main__":
    sys.exit(main())