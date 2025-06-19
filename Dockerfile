FROM continuumio/miniconda3

# Set working directory
WORKDIR .

# Install system build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy environment file and project files
COPY environment-deployment.yml .
COPY . .


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
CMD ["bash", "-c", "python -m segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m segment_tb_cxr/sample.csv segment_tb_cxr/yolov8/weights/yolov8.pt segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth segment_tb_cxr/sample_seg --binary_mask_threshold 0.5 segment_tb_cxr/sample_nnunet_preds.csv && python -m classification_tb_not_tb.generate_classification_results segment_tb_cxr/sample_nnunet_preds.csv classification_tb_not_tb/resnet_unet_configuration.json classification_tb_not_tb/cxr_segment.pt classification_results.csv"]
