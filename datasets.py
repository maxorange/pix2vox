import numpy as np
import glob, os, cv2
import util
import config

class Dataset(object):

    def __init__(self, args):
        self.index_in_epoch = 0
        self.examples = np.array(glob.glob(args.dataset_path))
        self.num_examples = len(self.examples)
        np.random.shuffle(self.examples)

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            np.random.shuffle(self.examples)
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.read_data(start, end)

    def read_voxel(self, filename):
        voxel = util.read_binvox(filename)
        return voxel

    def read_style(self, filename):
        style = np.load(filename)
        return style

    def read_image1(self, filename):
        image = cv2.imread(filename, 0).astype(np.float32) / 127.5 - 1
        image = np.expand_dims(image, -1)
        return image

    def read_image2(self, filename):
        image = cv2.imread(filename, 1).astype(np.float32) / 127.5 - 1
        return image

class Stage1(Dataset):

    def __init__(self, args):
        super(Stage1, self).__init__(args)

    def read_data(self, start, end):
        voxels = []
        images = []
        labels = []

        for binvox_path in self.examples[start:end]:
            head, tail = os.path.split(binvox_path)
            image_path = os.path.join(head, 'model_edge_{0}.png'.format(np.random.randint(5)))
            label_path = os.path.join(head, '../label.npy')

            voxel = self.read_voxel(binvox_path)
            image = self.read_image1(image_path)
            label = np.load(label_path)

            voxels.append(voxel)
            images.append(image)
            labels.append(label)

        return voxels, images, labels

class Stage2(Dataset):

    def __init__(self, args):
        super(Stage2, self).__init__(args)

    def read_data(self, start, end):
        styles = []
        images1 = []
        images2 = []
        labels = []

        for style_path in self.examples[start:end]:
            head, tail = os.path.split(style_path)
            index = np.random.randint(5)
            image1_path = os.path.join(head, 'model_edge_{0}.png'.format(index))
            image2_path = os.path.join(head, 'model_color_{0}.png'.format(index))
            label_path = os.path.join(head, '../label.npy')

            style = self.read_style(style_path)
            image1 = self.read_image1(image1_path)
            image2 = self.read_image2(image2_path)
            label = np.load(label_path)

            styles.append(style)
            images1.append(image1)
            images2.append(image2)
            labels.append(label)

        return styles, images1, images2, labels
