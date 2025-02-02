from matplotlib import pyplot as plt
import json
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision import tv_tensors

def show_sample(filename, root='data/', dpi='default', ax=plt) :

    if dpi != 'default' :
        dpi = f'/dpi{dpi}'
    image = read_image(root + f'images/{dpi}/' + filename + '.jpg')
    annotations = json.load(open(root + 'annotations.json'))
    annotation = annotations[filename]

    h, w = image.shape[1:]
    boxes = tv_tensors.BoundingBoxes(
        annotation['boxes'],
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=image.shape[-2:]
    )

    boxes[:, ::2] *= w
    boxes[:, 1::2] *= h
    
    image_to_plot =  draw_bounding_boxes(image, boxes, width=5, colors='red')

    # Plot the image
    ax.imshow(image_to_plot.permute(1, 2, 0))
    ax.axis('off')  # Hide the axis

def show_sample_from_dataset(dataset, idx, ax=plt) :

    img, target = dataset.__getitem__(idx)
    
    for annotation in target['boxes'] :
        xt, yt, xb, yb = annotation

        ax.plot([xt, xb, xb, xt, xt], [yt, yt, yb, yb, yt], 'r', alpha=0.2)
        ax.scatter(xt, yt, c='r', marker='o')
        ax.scatter(xb, yb, c='r', marker='o')

    # Transpose the dimensions to (H, W, C)
    img_toplot = img.permute(1, 2, 0).numpy()

    # Plot the image
    ax.imshow(img_toplot)
    ax.scatter(xt, yt, c='r', marker='o')
    ax.scatter(xb, yb, c='r', marker='o')
    ax.axis('off')  # Hide the axis