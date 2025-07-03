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

# Create the subdirectories for the files so the imports within python script work as expected FOR NNUNET
RUN mkdir -p fold_0/

# Copy environment file and project files
COPY environment-deployment.yml .

COPY classification_tb_not_tb/*.py .

COPY classification_tb_not_tb/lung_cxr_segmentation/segment_lung_cxr/training/resnet_unet_configuration.json .

COPY classification_tb_not_tb/lung_cxr_segmentation/segment_lung_cxr/data/weights/cxr_segment.pt .

COPY segment_tb_cxr/auxiliary/ensemble_nnunet_yolov8m.py .

COPY segment_tb_cxr/yolov8/weights/yolov8.pt .

# Copy required files for  nnunet model
COPY segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth  fold_0/nnunet.pth
COPY segment_tb_cxr/nnunet/weights/dataset.json .
COPY segment_tb_cxr/nnunet/weights/plans.json .

# Create environment
RUN conda env create -f environment-deployment.yml

# Activate env and set as default
SHELL ["conda", "run", "-n", "tbenv", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/tbenv/bin:$PATH

# Override yolov8 inference files with modified files , to return probabilities from yolov8 model which are not available in original implementation.
COPY segment_tb_cxr/auxiliary/yolov8/predict.py /opt/conda/envs/tbenv/lib/python3.10/site-packages/ultralytics/models/yolo/segment/predict.py
COPY segment_tb_cxr/auxiliary/yolov8/ops.py /opt/conda/envs/tbenv/lib/python3.10/site-packages/ultralytics/utils/ops.py
COPY segment_tb_cxr/auxiliary/yolov8/results.py /opt/conda/envs/tbenv/lib/python3.10/site-packages/ultralytics/engine/results.py

ENV EXTRA_ARGS=""

# Run app
CMD ["bash", "-c", "python -m ensemble_nnunet_yolov8m_tb_not_tb inputs yolov8.pt fold_0/nnunet.pth resnet_unet_configuration.json cxr_segment.pt inputs/sample_preds.csv $EXTRA_ARGS"]
