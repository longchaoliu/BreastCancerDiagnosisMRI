# Use the official TensorFlow base image (non-GPU version)
FROM tensorflow/tensorflow:2.17.0

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the GitHub repository
RUN git clone https://github.com/lkshrsch/BreastCancerDiagnosisMRI.git

# Set the working directory to the cloned repo
WORKDIR /BreastCancerDiagnosisMRI

# Install Python dependencies: TensorFlow and any other common packages
RUN pip3 install --upgrade pip && \
    pip3 install tensorflow==2.17.0 nibabel scikit-image numpy pandas scikit-learn matplotlib jupyter


# Run the prediction script on container start
# CMD ["python", "Make_Prediction_on_MRI.py"]


# Expose port 8888 for Jupyter Notebook
# EXPOSE 8888

# Command to run Jupyter Notebook when the container starts
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
