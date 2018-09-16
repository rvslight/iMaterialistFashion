from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import json
import urllib3
import multiprocessing
import numpy
import shutil

from PIL import Image, ImageOps
from tqdm import tqdm
from urllib3.util import Retry

# numpy.random.seed(1)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def add_pad_image(image):
    # img = cv2.imread(image, 1)
    w, h = image.size
    if h > w:
        max_size = h
    else:
        max_size = w
    pad_h = int((max_size - h) / 2)
    pad_w = int((max_size - w) / 2)
    padding = (pad_w,pad_h,pad_w,pad_h)
    pad_image = ImageOps.expand(image, padding)
    return pad_image

def download_image(fnames_and_urls):
    """
    download image and save its with 90% quality as JPG format
    skip image downloading if image already exists at given path
    :param fnames_and_urls: tuple containing absolute path and url of image
    """
    fname, url = fnames_and_urls
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3, backoff_factor=0.3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb = add_pad_image(image_rgb)
        image_rgb.save(fname, format='JPEG', quality=90)

def parse_dataset(_dataset, _outdir, _num):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _outdir: output directory where data will be saved
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    _fnames_urls = []
    with open(dataset, 'r') as f:
        data = json.load(f)
        if (num):
            ids = numpy.random.choice(len(data["images"]), num, replace=False)
            print("Downloading {} random images out of {}:".format(num,len(data["images"])))
            with open(outdir + 'label.json', 'w') as outfile:
                newlist = {}
                newlist["images"] = []
                newlist["annotations"] = []
                for id in ids:
                    url = data["images"][id]["url"]
                    fname = os.path.join(outdir, "{}.jpg".format(data["images"][id]["imageId"]))
                    _fnames_urls.append((fname, url))
                    newlist["annotations"].append(data["annotations"][id])
                json.dump(newlist, outfile)
        else:
            for image in data["images"]:
                url = image["url"]
                fname = os.path.join(outdir, "{}.jpg".format(image["imageId"]))
                _fnames_urls.append((fname, url))
    return _fnames_urls

#make label txt

def save_label_txt(fnames_and_labels):
    """
    download image and save its with 90% quality as JPG format
    skip image downloading if image already exists at given path
    :param fnames_and_urls: tuple containing absolute path and url of image
    """

    fname, labels = fnames_and_labels
    labels = list(map(int, labels))  # change to int
    labels.sort()  # sorting
    if not os.path.exists(fname):
        # save txt as fname.. txt
        f = open(fname, 'w')
        label_txt = '\n'.join(str(label) for label in labels)
        f.write(label_txt)
        f.close()

        # http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3, backoff_factor=0.3))
        # response = http.request("GET", url)
        # image = Image.open(io.BytesIO(response.data))
        # image_rgb = image.convert("RGB")
        # image_rgb.save(fname, format='JPEG', quality=90)

def parse_dataset_label(_dataset, _outdir):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _outdir: output directory where data will be saved
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    _fnames_urls = []
    with open(dataset, 'r') as f:
        data = json.load(f)

        for annotations in data["annotations"]:
            labelId = annotations["labelId"]
            fname = os.path.join(outdir, "{}.jpg.txt".format(annotations["imageId"]))
            _fnames_urls.append((fname, labelId))
    return _fnames_urls

if __name__ == '__main__':
    if len(sys.argv) == 4:
        dataset, outdir, num = sys.argv[1], sys.argv[2], int(sys.argv[3])
    elif len(sys.argv) == 3:
        dataset, outdir, num = sys.argv[1], sys.argv[2], 0
    else:
        print("Usage: python downloader.py file.json path/to/download (max_entries)")
        sys.exit(0)

    if os.path.exists(outdir):
        # remove file and folder
        shutil.rmtree(outdir)
        shutil.rmtree('./label')

    if not os.path.exists(outdir):

        os.makedirs(outdir)



    # parse json dataset file
    fnames_urls = parse_dataset(dataset, outdir, num)

    # download data
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(fnames_urls)) as progress_bar:
        for _ in pool.imap_unordered(download_image, fnames_urls):
            progress_bar.update(1)

    # move foldeer
    source = './image/'
    dest1 = './image/folder/'

    os.makedirs(dest1)
    files = os.listdir(source)

    for f in files:
        shutil.move(source + f, dest1)

    #make label_txt_file
    dataset = "imagelabel.json"
    outdir = "label"

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fnames_and_labels = parse_dataset_label(dataset, outdir)
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(fnames_and_labels)) as progress_bar:
        for _ in pool.imap_unordered(save_label_txt, fnames_and_labels):
            progress_bar.update(1)
    total_label = []
    for i in range (len(fnames_and_labels)):
        total_label.append(fnames_and_labels[i][1])
    total_label = [val for sublist in total_label for val in sublist] # flatten
    total_label = list(set(total_label)) # set
    total_label = list(map(int, total_label)) # change to int
    total_label.sort() #sorting
    #make total label list
    f = open('../labels.txt', 'w')

    label_txt = '\n'.join(str(label) for label in total_label)
    f.write(label_txt)
    f.close()

    sys.exit(1)
