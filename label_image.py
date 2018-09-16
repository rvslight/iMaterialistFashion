import tensorflow as tf
import sys
import numpy as np
import os

def submission_format(ids, labels):
    # transform final predictions into correct submission format
    ids = np.array(ids)
    labels = np.array(labels)

    [r] = labels.shape
    with open('submission.csv', 'w') as f:
        f.write("image_id,label_id")
        f.write('\n')
        for i in range(r):
            l = labels[i]
            data = '{},{}'.format(ids[i], l)
            f.write(data)
            f.write('\n')


# change this as you see fit
image_path = sys.argv[1]

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("labels.txt")]

ids_list = []
labels_list = []



# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')



with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    files = {}
    for number, filename in enumerate(sorted(os.listdir(image_path)), start=0):
        files[number] = filename

    for i in range(len(files)):

        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path+files[i], 'rb').read()

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        resultString = ""
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > 0.7:
                print('%s (score = %.5f)' % (human_string, score))
                if resultString == "":
                    resultString = human_string
                else :
                    resultString = resultString + " " + human_string
        ids_list.append(files[i].split(".")[0])
        print(os.path.splitext(image_path)[0].split("/")[-1])
        print(resultString)
        labels_list.append(resultString)


submission_format(ids_list, labels_list)
