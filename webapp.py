import argparse, cv2, json, os, signal
import labels
import numpy as np
import tornado.ioloop
import tornado.web
import tornado.httpserver
import util
from model import sgan

class Dataset(object):
    def __init__(self, args):
        self.labels = labels.labels[args.dataset]
        self.n_labels = len(self.labels.items())

class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def get(self):
        labels = self.dataset.labels
        labels = sorted(labels.items(), key=lambda x: x[0])
        self.render('volume.html', labels=labels)

class ImageHandler(tornado.web.RequestHandler):
    def initialize(self, model, args):
        self.model = model
        self.npx = args.npx
        self.z = np.random.uniform(-1, 1, size=(args.batch_size, model.nz))

    def post(self):
        image = self.__decode_image()
        label = self.__decode_label()
        volume = self.model.generate(image, self.z, label)
        mesh = util.extract_mesh(np.squeeze(volume))
        self.write(json.dumps(mesh))
        # self.write(json.dumps(dict(hello=1, world=2)))

    def __decode_image(self):
        data = self.get_argument('image')
        data = json.loads(data)
        npx = self.npx
        depthmap = np.zeros((npx, npx, 1), np.uint8)
        edged = np.zeros((npx, npx, 1), np.uint8)
        for pnts, c, w, mode in zip(data['strokes'], data['colors'], data['widths'], data['modes']):
            c = c.lstrip('#')
            c = int(c[:2], 16)
            w = int(w*npx)
            for i in range(len(pnts)-1):
                pnt1 = (int(pnts[i][0]*npx), int(pnts[i][1]*npx))
                pnt2 = (int(pnts[i+1][0]*npx), int(pnts[i+1][1]*npx))
                if mode == 'depthmap':
                    cv2.line(depthmap, pnt1, pnt2, c, w)
                else:
                    cv2.line(edged, pnt1, pnt2, 1, w)
        image = np.concatenate((depthmap, edged), -1)
        return np.expand_dims(edged, 0)

    def __decode_label(self):
        label = self.get_argument('label')
        label = json.loads(label)
        label = np.expand_dims(label, 0)
        return label

class Application(tornado.web.Application):
    def __init__(self, dataset, model, args):
        handlers = []
        handlers.append((r'/', IndexHandler, dict(dataset=dataset, model=model)))
        handlers.append((r'/image', ImageHandler, dict(model=model, args=args)))

        settings = {}
        settings['template_path'] = os.path.join(os.getcwd(), 'templates')
        settings['static_path'] = os.path.join(os.getcwd(), 'static')
        settings['debug'] = True
        settings['autoreload'] = False

        super(Application, self).__init__(handlers, **settings)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nsf', type=int, default=4, help='encoded pixel size of generator')
    parser.add_argument('--npx', type=int, default=64, help='output pixel size')
    parser.add_argument('--nvx', type=int, default=32, help='output voxel size')
    parser.add_argument('--params_path', type=str, default='params/sgan_model.ckpt')
    parser.add_argument('--dataset', type=str, default='shapenetcore-v1')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = Dataset(args)
    model = sgan.Model(args.params_path)

    application = Application(dataset, model, args)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    print 'start server'

    def sig_handler(sig, frame):
        tornado.ioloop.IOLoop.instance().add_callback(shutdown)

    def shutdown():
        model.sess.close()
        http_server.stop()
        tornado.ioloop.IOLoop.instance().stop()

    signal.signal(signal.SIGINT, sig_handler)
    tornado.ioloop.IOLoop.instance().start()
