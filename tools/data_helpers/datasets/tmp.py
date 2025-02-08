# importing prerequisites
import sys
import requests
import tarfile
import json
import numpy as np
import pdf2image
from os import path
from PIL import Image
from PIL import ImageFont, ImageDraw
from glob import glob
from matplotlib import pyplot as plt
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
from IPython.core.display import display, HTML
import pdb
import copy


# Verifying the file was extracted properly
data_path = "examples/"
path.exists(data_path)

# Define color code
colors = [(255, 0, 0),(0, 255, 0)]
categories = ["table", "cell"]


# Function to viz the annotation
def markup(image, annotations, pdf_height):
    ''' Draws the segmentation, bounding box, and label of each annotation
    '''
    draw = ImageDraw.Draw(image, 'RGBA')
    for annotation in annotations:
        # Draw bbox
        orig_annotation = copy.copy(annotation['bbox'])
        annotation['bbox'][3] = pdf_height-orig_annotation[1]
        annotation['bbox'][1] = pdf_height-orig_annotation[3]
        draw.rectangle(
            (annotation['bbox'][0],
             annotation['bbox'][1],
             annotation['bbox'][2],
             annotation['bbox'][3]),
            outline=colors[annotation['category_id'] - 1] + (255,),
            width=2
        )
        # Draw label
        w, h = draw.textsize(text=categories[annotation['category_id'] - 1])
        if annotation['bbox'][3] < h:
            draw.rectangle(
                (annotation['bbox'][2],
                 annotation['bbox'][1],
                 annotation['bbox'][2] + w,
                 annotation['bbox'][1] + h),
                fill=(64, 64, 64, 255)
            )
            draw.text(
                (annotation['bbox'][2],
                 annotation['bbox'][1]),
                text=categories[annotation['category_id'] - 1],
                fill=(255, 255, 255, 255)
            )
        else:
            draw.rectangle(
                (annotation['bbox'][0]-w,
                 annotation['bbox'][1]-h,
                 annotation['bbox'][0],
                 annotation['bbox'][1]),
                fill=(64, 64, 64, 255)
            )
            draw.text(
                (annotation['bbox'][0]-w,
                 annotation['bbox'][1]-h),
                text=categories[annotation['category_id'] - 1],
                fill=(255, 255, 255, 255)
            )
    return np.array(image)


# Parse the JSON file and read all the images and labels
with open('examples/FinTabNet_1.0.0_table_example.jsonl', 'r') as fp:
    images = {}
    for line in fp:
        sample = json.loads(line)
        # Index images
        if sample['filename'] in images:
            annotations = images[sample['filename']]["annotations"]
            html = images[sample['filename']]["html"]
        else:
            annotations = []
            html = ""
        for t, token in enumerate(sample["html"]["cells"]):
            if "bbox" in token:
                annotations.append({"category_id":2, "bbox": token["bbox"]})
        #Build html table
        cnt = 0
        for t, token in enumerate(sample["html"]["structure"]["tokens"]):
            html += token
            if token=="<td>":
                html += "".join(sample["html"]["cells"][cnt]["tokens"])
                cnt += 1
        annotations.append({"category_id": 1, "bbox": sample["bbox"]})
        images[sample['filename']] = {'filepath': 'examples/pdf/' + sample["filename"], 'html': html, 'annotations': annotations}

# Visualize annotations and print HTML tables
import matplotlib
for i, (filename, image) in enumerate(images.items()):
    pdf_page = PdfFileReader(open(image["filepath"], 'rb')).getPage(0)
    pdf_shape = pdf_page.mediaBox
    pdf_height = pdf_shape[3]-pdf_shape[1]
    pdf_width = pdf_shape[2]-pdf_shape[0]
    converted_images = convert_from_path(image["filepath"], size=(pdf_width, pdf_height))
    img = converted_images[0]
    print("Table HTML for page #{}".format(i))
    display(HTML(image['html']))
    plt.figure()
    plt.imshow(markup(img, image['annotations'], pdf_height))
    plt.title("Page # {}".format(i))
    plt.axis('off')