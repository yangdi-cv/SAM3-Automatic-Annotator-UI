# SAM3 Auto Annotator
▶️ **[Tutorial video](https://youtu.be/7OUmLD2FzPE)**

An image annotation tool integrated with SAM3 (Segment Anything Model 3) for semantic segmentation, object detection, and instance segmentation.

![Demo](demo.gif)

## Features

- **Three Annotation Modes**:
  - **Detection**: Bounding boxes only.
  - **Instance**: Polygons with bounding boxes (default).
  - **Semantic**: Fused connected instances with conflict resolution.
- **AI-Assisted Segmentation**: Automatic instance segmentation using SAM3 model.
- **Interactive Editing**:
  - **Polygon**: Drag vertices, add/remove points.
  - **BBox**: Resize using corner handles.
- **Undo/Redo**: Support for undoing last operations (vertex move, delete, create, etc.).
- **Multi-class Labeling**: Support for multiple categories with custom colors.
- **Batch Processing**: Navigate through image directories efficiently.
- **Annotation Format**: Export annotations in LabelMe-compatible JSON format with extended metadata.

## Requirements

```bash
# Create a conda environment
conda create -n auto-label python=3.10
conda activate auto-label

# install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# install Transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e ".[torch]"
cd ..

# install OpenCV, PyQt5, Pillow
pip install opencv-python PyQt5 Pillow
```

## Installation

1. Clone the repository
2. Sign in to your Hugging Face account and request free access to the SAM3 model from the official [Hugging Face repository](https://huggingface.co/facebook/sam3/tree/main)
3. Download the SAM3 model weights and place them in a folder named `sam3_weights`
4. Install dependencies (see Requirements above)

## Usage

### Launch Application
```bash
python auto-label.py
```

### Keyboard Shortcuts

- **A**: Previous image
- **D**: Next image
- **S**: Save annotations and exit edit mode
- **Delete**: Delete selected annotation
- **Ctrl+Z**: Undo last operation (up to 3 steps)
- **Alt+Drag**: Pan canvas (ignores masks)

### Workflow

1. **Setup**
   - Select input folder containing images
   - Select output folder for JSON annotations
   - Add labels with custom colors
   - **Select Annotation Mode**: Choose between Detection, Instance, or Semantic.

2. **Auto-Segmentation (RUN)**
   - Click "RUN" to execute SAM3 model for current labels
   - The model will generate masks/bboxes based on the selected mode

3. **Manual Annotation (Create Mode)**
   - Switch to "Create" mode
   - **Left click**: Add polygon vertices
   - **Right click**: Undo last vertex
   - **Click near start point or Enter**: Complete polygon
   - **ESC**: Cancel current polygon

4. **Edit Mode**
   - Double-click an annotation or switch to "Edit" mode to modify.
   - **Polygons**:
     - Drag vertices to adjust shape
     - **Ctrl + Left click** on edge: Insert new vertex
     - **Right click** on vertex: Delete vertex (min 3 vertices)
   - **Bounding Boxes**:
     - Drag corner handles to resize
   - **General**:
     - Click label in list: Change category of selected annotation

5. **Save**
   - Press "S" or click "Save" button to export annotations to JSON

## Output Format

Annotations are saved in a JSON format compatible with LabelMe, extended with additional metadata:

```json
{
  "imagePath": "image.jpg",
  "imageHeight": 1080,
  "imageWidth": 1920,
  "shapes": [
    {
      "label": "car",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "instance_polygon",
      "bbox": [x_min, y_min, x_max, y_max],
      "score": 0.95,
      "instance_id": 1,
      "area": 1500.5
    },
    {
      "label": "person",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "detection_bbox",
      "bbox": [x_min, y_min, x_max, y_max],
      "score": 0.88
    },
    {
      "label": "road",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "semantic_polygon",
      "component_id": 1,
      "area": 5000.0
    }
  ]
}
```

*Note: `shape_type` can be `detection_bbox`, `instance_polygon`, or `semantic_polygon`.*

## Configuration

- **Model loading**: Ensure SAM3 weights are correctly placed in `sam3_weights` folder.
- **Persistence**: Input/output paths, labels, and last used annotation mode are saved in `config.json`.

## Acknowledgements
We sincerely thank the [SAM3](https://ai.meta.com/sam3/) team for open-sourcing their work.
