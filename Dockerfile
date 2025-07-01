FROM continuumio/miniconda3

# Set working directory
WORKDIR /

# Install system build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0

# Create the subdirectories for the files so the imports within python script work as expected
RUN mkdir -p classification_tb_not_tb

RUN mkdir -p segment_tb_cxr/auxiliary

RUN mkdir -p segment_tb_cxr/yolov8/weights

RUN mkdir -p segment_tb_cxr/nnunet/weights/fold_0/

# Copy environment file and project files
COPY environment-deployment.yml .

COPY classification_tb_not_tb/ classification_tb_not_tb/

COPY segment_tb_cxr/auxiliary/ensemble_nnunet_yolov8m.py segment_tb_cxr/auxiliary/ensemble_nnunet_yolov8m.py

COPY segment_tb_cxr/auxiliary/ensemble_nnunet_yolov8m_tb_not_tb.py segment_tb_cxr/auxiliary/ensemble_nnunet_yolov8m_tb_not_tb.py

COPY segment_tb_cxr/auxiliary/compute_probability_of_TB_from_segmentation.py segment_tb_cxr/auxiliary/compute_probability_of_TB_from_segmentation.py

COPY segment_tb_cxr/yolov8/weights/yolov8.pt segment_tb_cxr/yolov8/weights/yolov8.pt

COPY segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth  segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth

COPY segment_tb_cxr/nnunet/weights/dataset.json segment_tb_cxr/nnunet/weights/dataset.json

COPY segment_tb_cxr/nnunet/weights/plans.json segment_tb_cxr/nnunet/weights/plans.json

# Create environment
RUN conda env create -f environment-deployment.yml

# Activate env and set as default
SHELL ["conda", "run", "-n", "tbenv", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/tbenv/bin:$PATH

# Copy predict.py override
COPY segment_tb_cxr/auxiliary/yolov8/predict.py /opt/conda/envs/tbenv/lib/python3.10/site-packages/ultralytics/models/yolo/segment/predict.py

# Copy ops.py override
COPY segment_tb_cxr/auxiliary/yolov8/ops.py /opt/conda/envs/tbenv/lib/python3.10/site-packages/ultralytics/utils/ops.py

# Copy results.py override
COPY segment_tb_cxr/auxiliary/yolov8/results.py /opt/conda/envs/tbenv/lib/python3.10/site-packages/ultralytics/engine/results.py

# Optional: expose port
EXPOSE 8000

# Run app
CMD ["bash", "-c", "python -m classification_tb_not_tb.ensemble_nnunet_yolov8m_tb_not_tb sample_inputs segment_tb_cxr/yolov8/weights/yolov8.pt segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth classification_tb_not_tb/lung_cxr_segmentation/segment_lung_cxr/training/resnet_unet_configuration.json classification_tb_not_tb/lung_cxr_segmentation/segment_lung_cxr/data/weights/cxr_segment.pt sample_preds.csv"]
