from pdf2image import convert_from_path
import sys
import json
from PIL import Image
import os

# Specify the path to the input PDF folder
raw_data_path = 'data/raw'

def preprocess_PDF2Images(dpi : int, output_folder : str = 'data/images', input_folder=raw_data_path):
    """
    Convert PDF files in the specified directory to JPG images.

    Parameters:
    dpi (int): The resolution in DPI (dots per inch) for the output images.

    Returns:
    None
    """

    if dpi :
        # Define the output folder where images will be saved
        output_folder = f'{output_folder}/dpi{dpi}'
    else :
        output_folder = f'{output_folder}/default'

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the raw data path
    for i_doc, file in enumerate(os.listdir(input_folder)):
        # Check if the file is a PDF
        if file.endswith('.pdf'):
            # Extract the filename without the extension
            filename = file.split('.')[0]

            if dpi :
                # Convert the PDF to a list of images
                images = convert_from_path(os.path.join(input_folder, file), dpi=dpi)
            else :
                images = convert_from_path(os.path.join(input_folder, file))

            # Iterate over each image in the converted list
            for i, image in enumerate(images):
                # Define the path for the output image
                image_path = os.path.join(output_folder, f'{filename}_p{i + 1}.jpg')

                # Save the image as a JPEG file
                image.save(image_path, 'JPEG')

            # Print the progress of the conversion
            print(f'Converted {len(images)} pages to JPG images. --- {100 * (i_doc + 1) / len(os.listdir(raw_data_path)):.0f}%')

def get_box_boundaries(region, size):
        
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        width = region['shape_attributes']['width']
        height = region['shape_attributes']['height']

        xt      = x / size[0] # relative position of center x of rect
        yt      = y / size[1] # relative position of center y of rect
        xb  = (x + width) / size[0]
        yb = (y + height) / size[1]

        return xt, yt, xb, yb

def convert_raw_annotations(root):
    """
    Convert raw annotations from a JSON file to a structured format.
    """
    with open(root + 'VIA/annotations_raw.json') as file:
        dict = json.load(file)
        
        # try:        
        #     namesFile = sys.argv[1:][1]
        #     names = open(namesFile).read().split('\n')
        # except IndexError:
        #     print >> sys.stderr, "names file's missing from argument.\n\tnamesFile = sys.argv[1:][1]\nIndexError: list index out of range"

        res_dict = {}

        for key in dict.keys():
            data = dict[key]

            imageName = data['filename']
            filename = imageName.rsplit('.', 1)[0]
            
            labels = []
            boxes = []
            
            regions = data['regions']

            try:        
                img = Image.open(root + 'images/default/' + imageName)
            except IOError:
                print >> sys.stderr, "No such file" , imageName

            content = ""
            for region in regions:
                labels.append(1) #get_object_class(region, imageName, names)
                annotation = get_box_boundaries(region, img.size)
                boxes.append(annotation)
            
            res_dict[filename] = {
                'labels': labels,
                'boxes': boxes,
                #'is_new' : True
                }

        with open("data/annotations.json", "w") as outFile:
            json.dump(res_dict, outFile)