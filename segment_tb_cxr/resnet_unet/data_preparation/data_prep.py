import glob
import SimpleITK as sitk
import pathlib
import argparse
import json
import pandas as pd


def filter_df(df, unwanted_findings):
     for i, (findings_list, boxes, scores_list) in enumerate(zip(df['Predicted Disease for Each ROI'], df['Locations of Boundary for Each ROI'], df['PredictedScores'])):
         # Identify indices of unwanted findings
         indices_to_remove = [index for index, finding in enumerate(findings_list) if finding in unwanted_findings]

         # Filter findings, boxes and  scores based on indices
         filtered_findings = [finding for index, finding in enumerate(findings_list) if index not in indices_to_remove]
         filtered_boxes = [box for index, box in enumerate(boxes) if index not in indices_to_remove]
         filtered_scores = [score for index, score in enumerate(scores_list) if index not in indices_to_remove]

         # Update DataFrame
         df.loc[i, 'Predicted Disease for Each ROI'] = str(filtered_findings)
         df.loc[i, 'Locations of Boundary for Each ROI'] = str(filtered_boxes)       
         df.loc[i, 'PredictedScores'] =  str(filtered_scores)
         
     df['Predicted Disease for Each ROI'] = df['Predicted Disease for Each ROI'].apply(lambda x: eval(x))
     df['Locations of Boundary for Each ROI'] = df['Locations of Boundary for Each ROI'].apply(lambda x: eval(x))
     df['PredictedScores'] = df['PredictedScores'].apply(lambda x: eval(x))         

     return df

def find_full_path(data_root,row):
     if len(list((data_root/pathlib.Path(row['PatientID'])).rglob(str(row['Filename'].split(".jpeg")[0]+".dcm")))) > 0:
         return str(list((data_root/pathlib.Path(row['PatientID'])).rglob(str(row['Filename'].split(".jpeg")[0]+".dcm")))[0])
     else:
         return str(list((data_root/pathlib.Path(row['PatientID'])).rglob(str(row['Filename'].split(".jpeg")[0])))[0])
         
def replace_finding_names(findings):
     modified_findings = []
     for finding in findings:
         if finding in ['Interstitial changes in the lungs' , 'Pulmonary Mesenchyme Denaturation'] :
              finding = 'Interstitial changes in the lungs/Pulmonary Mesenchyme Denaturation'
              modified_findings.append(finding)
         elif finding in ['Isolated Tracheal or Bronchial Tuberculosis' ,'Bronchial tuberculosis'] :
              finding = 'Isolated Tracheal or Bronchial Tuberculosis/Bronchial Tuberculosis'
              modified_findings.append(finding)
         else:
              modified_findings.append(finding)
     return modified_findings
           
df = pd.read_csv('TB_Portals_labeled20231121.csv')

all_findings = pd.read_csv('66_abnormalities_disease_and_manifestation.csv')[['SIFTS Abnormality Name','used_or_not_used']]
used_findings = all_findings[all_findings['used_or_not_used'] == 'used']['SIFTS Abnormality Name']
notused_findings = all_findings[all_findings['used_or_not_used'] == 'not_used']['SIFTS Abnormality Name'].tolist()

finding_to_label_value_dict = dict(zip(used_findings, range(1, len(used_findings) + 1)))

df['Locations of Boundary for Each ROI'] = df['Locations of Boundary for Each ROI'].apply(lambda x: eval(x))
df['PredictedScores'] = df['PredictedScores'].apply(lambda x: eval(x))
df['Predicted Disease for Each ROI'] = df['Predicted Disease for Each ROI'].apply(lambda x: replace_finding_names(eval(x)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Data preparation script")

    parser.add_argument(
        "input_",
        type=pathlib.Path,
        help="Input JSON directory containing json files containing relative information\
             about Chest X Ray and the matching reference label filename",
    )

    parser.add_argument(
        "input_cxr_dir",
        type=pathlib.Path,
        help="Input Chest X Ray mask directory containing annotations for covid19 dataset",
    )

    parser.add_argument(
        "input_reference_mask_dir",
        type=pathlib.Path,
        help="Input reference mask directory containing annotations for covid19 dataset",
    )

    parser.add_argument(
        "output_csv_filename",
        type=pathlib.Path,
        help="Output CSV filename containing paths for Chest X Ray and corresponding \
              reference label in 'cxr_file' and 'ref_seg_file'  columns respectively. ",
    )

    args = parser.parse_args()

    cxr_files, seg_files = _get_all_files(
        args.input_json_dir,
        args.input_cxr_dir,
        args.input_reference_mask_dir,
        args.output_csv_filename,
    )

    all_files = pd.DataFrame({"cxr_file": cxr_files, "seg_file": seg_files})
    all_files.to_csv(args.output_csv_filename, index=False)


if __name__ == "__main__":
    main()
