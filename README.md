# Arrayxtract

This project aims at giving the tools to extract arrays or tables from scanned PDF files.

**Current state:**
- extracts croped tables (.jpg),
- saves relative positions of tables on pages.

**Future iterations:**
- implement a corner-type detection model, therefore making it possible to detect table's shape,
- implement an OCR module to reproduce tables in structured formats such as csv or xlsx.

**Faced issues:**

Please be aware that the solutions to these issues were not tested yet. They are suggestions.
- Titles sometimes mistaken for tables (and vice-versa for 1-row tables):
    - (solution) add a second label for framed titles, to help the model distinguish both,
    - (solution) implement a corner-type detection model (chich would detect framed titles) and add a simple NN classifier on top of this.
- The croped table sometimes hides borders,
    - (solution) tweak the loss function to add a malus to bounding boxes predictions that are smaller than the ground truth boxes,
    - (solution) make each detected table bigger by XXX pixels (tbd with threshold).

**Models weights download links:**

- [Resnet50 backbone](https://drive.google.com/file/d/1yumVmEUBlykwt-ewLWUb57_TzlbwWhqx/view?usp=sharing)
- [MobileNetV3 backbone](https://drive.google.com/file/d/1gAJGRCypK2aelq3gi0JekaTbmaym8UbH/view?usp=sharing)

## Project structure

### Formatting
- data
    - raw --> *contains raw pdf used for training*
    - VIA --> *annotation files*
- models --> *model weights, to be downloaded*
- notebooks --> *multiple notebooks containing steps of the training pipeline & one inference notebook*
- results --> *losses history of the different models*
- scipts --> *.py files containing necessary functions, classes and methods for the project*

### Preprocessing

- Raw files (pdf) are in the *data/raw* folder.
- They are to be processed with the *notebooks/preprocess.ipynb* notebook, whose main functions and methods lie in *scripts/data_preprocessing.ipynb*, each page is converted to an image.
- They were manually annotated using [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/).
- During the training, the data run though a set of transformations, as described in the *scripts/dataset.py* file.

### Model and training
- Tested models consist of Fast-RCNNs finetuned to output two classes (instead of 80), with three different backbones (see *notebooks/train_model.ipynb*), trained on 240 augmented images.
- Models are trained for 10 epochs.
- Performances on different losses values (average, classification and box regression).

### Model selection 
- *Resnet50* backbone clearly offers best performance than the *MobileNetV3* (see plots in *notebooks/visualize_results.ipynb* notebook).
- Visual inspection of prediction also shows that the model performs better.
- This however comes with twice more parameters, leading to more computation time.

### Inference

Put the pdf file to be processed in *data/inference*, execute the *notebooks/compute_pdf/ipynb* file. Results will be stored in *data/inference/output*.