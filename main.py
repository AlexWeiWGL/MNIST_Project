import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request,send_file
from mnist import model
from PIL import Image
from cassandra.cluster import Cluster
import datetime


x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def convolutional(input):
    return sess.run(y, feed_dict={x:input, keep_prob:1.0}).flatten().tolist()

def cassandraconnect(uploadtime, name, prediction):
    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
    session = cluster.connect("mnist")
    session.execute("""INSERT INTO pictures(upload_time, picture_name, prediction_number)
                    VALUES (%s, %s, %s)""", (uploadtime, name, prediction))
    cluster.shutdown()


app = Flask(__name__)


@app.route('/mnist/prediction', methods=['GET','POST'])
def mnist():
    if request.method == 'POST':
        max = 0
        max_index = 0
        time = datetime.datetime.now().strftime('%Y-%m-%d %T')
        file = request.files.get('photo')
        file.save(file.filename)
        input = np.array(Image.open(file.filename), dtype=np.uint8).reshape(1, 784)
        output = convolutional(input)
        for i in range(0, len(output)):
            if output[i] >= max:
                max = output[i]
                max_index = i
        cassandraconnect(time, file.filename, max_index)
        return render_template('return.html', result=max_index)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8000)

