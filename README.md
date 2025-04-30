# MobileSAM Core ML Converter

This project converts the MobileSAM (Segment Anything Model) to Core ML format for use in iOS and macOS applications.

## Overview

The converter script transforms MobileSAM into two Core ML models:
- `MobileSAMEmbedding.mlpackage`: Handles image encoding
- `MobileSAMMaskDecoder.mlpackage`: Processes embeddings to generate segmentation masks

## Requirements

- Python 3.11+
- PyTorch 2.x
- coremltools
- MobileSAM dependencies

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd MobileSAM_project
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Unix/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the conversion script:
```bash
python convert_mobilesam_to_coreml.py
```

The converted models will be saved in the `coreml_models` directory.

## Model Specifications

- Compute Precision: FP16
- Compute Units: CPU, GPU, Neural Engine
- Input Shapes:
  - Embedding Model: (1, 3, 1024, 1024) for RGB images
  - Mask Decoder: Takes image embeddings, point coordinates, and point labels

## License

This project is licensed under the terms specified in the MobileSAM repository. 