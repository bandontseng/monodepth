# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
import numpy as np
import argparse
import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt

from monodepth_model import monodepth_parameters, MonodepthModel

# import for HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial
from io import BytesIO
import json
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

parser = argparse.ArgumentParser(
    description='Monodepth TensorFlow implementation.')

parser.add_argument(
    '--encoder',
    type=str,
    help='type of encoder, vgg or resnet50',
    default='vgg')
parser.add_argument(
    '--checkpoint_path',
    type=str,
    help='path to a specific checkpoint to load',
    default="model_kitti/model_kitti")
parser.add_argument(
    '--input_height',
    type=int,
    help='input height',
    default=256)
parser.add_argument('--input_width', type=int, help='input width', default=512)

args = parser.parse_args()


class MonoDepthModel:

    def post_process_disparity(self, disp):
        _, h, w = disp.shape
        l_disp = disp[0, :, :]
        r_disp = np.fliplr(disp[1, :, :])
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
        r_mask = np.fliplr(l_mask)
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

    def __init__(self, params):
        """Test function."""

        self._left = tf.placeholder(
            tf.float32, [
                2, args.input_height, args.input_width, 3])
        self._model = MonodepthModel(params, "test", self._left, None)

        """
        for fname in os.listdir(args.image_path):
            input_image = scipy.misc.imread(args.image_path, mode="RGB")
            original_height, original_width, num_channels = input_image.shape
            input_image = scipy.misc.imresize(
                input_image, [
                    args.input_height, args.input_width], interp='lanczos')
            input_image = input_image.astype(np.float32) / 255
            input_images = np.stack((input_image, np.fliplr(input_image)), 0)
        """

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        self._sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        # coordinator = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # RESTORE
        restore_path = args.checkpoint_path.split(".")[0]
        train_saver.restore(self._sess, restore_path)

        """
        for fname in os.listdir(args.image_path):
            input_image = scipy.misc.imread(args.image_path + "/" + fname, mode="RGB")
            original_height, original_width, num_channels = input_image.shape
            input_image = scipy.misc.imresize(
                input_image, [
                    args.input_height, args.input_width], interp='lanczos')
            input_image = input_image.astype(np.float32) / 255
            input_images = np.stack((input_image, np.fliplr(input_image)), 0)

            disp = sess.run(self._model.disp_left_est[0], feed_dict={self._left: input_images})
            disp_pp = self.post_process_disparity(disp.squeeze()).astype(np.float32)

            # output_directory = os.path.dirname(args.image_path)
            output_directory = args.output_path
            output_name = os.path.splitext(fname)[0]

            np.save(
                os.path.join(
                    output_directory,
                    "{}.npy".format(output_name)),
                disp_pp)
            disp_to_img = scipy.misc.imresize(
                disp_pp.squeeze(), [
                    original_height, original_width])
            plt.imsave(
                os.path.join(
                    output_directory,
                    "{}.png".format(output_name)),
                disp_to_img,
                cmap='plasma')

        print('done!')
        """

    def __call__(self, input_image):
        original_height, original_width, num_channels = input_image.shape
        input_image = scipy.misc.imresize(
            input_image, [
                args.input_height, args.input_width], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)

        disp = self._sess.run(self._model.disp_left_est[0], feed_dict={self._left: input_images})
        disp_pp = self.post_process_disparity(disp.squeeze()).astype(np.float32)

        disp_to_img = scipy.misc.imresize(
            disp_pp.squeeze(), [
                original_height, original_width])
        return disp_to_img


class MonoDepthHTTPRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, model, *args, **kwargs):
        self._model = model
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()
        response = BytesIO()
        recvData = json.loads(body.decode("ASCII"))
        inputFrame = np.frombuffer(base64.decodebytes(recvData["frame"].encode("ASCII")), dtype=np.uint8)
        inputFrame = inputFrame.reshape(recvData["shape"])

        outputFrame = self._model(inputFrame)

        resultDict = {"shape": outputFrame.shape, "frame": base64.b64encode(outputFrame).decode("ASCII")}

        if "extData" in recvData:
            resultDict["extData"] = recvData["extData"]

        resultData = json.dumps(resultDict)
        response.write(resultData.encode("ASCII"))
        self.wfile.write(response.getvalue())


def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    model = MonoDepthModel(params)

    server = HTTPServer(('', 8000), partial(MonoDepthHTTPRequestHandler, model))
    server.serve_forever()

if __name__ == '__main__':
    # tf.app.run()
    main(None)
