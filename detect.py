import sys
import time

from PIL import Image, ImageDraw
from darknet import Darknet
from utils import *


def detect_with_model(model, image, verbose=True, use_cuda=0):
    if verbose:
        model.print_network()

    if use_cuda:
        model.cuda()

    img = Image.open(image).convert('RGB')
    sized = img.resize((model.width, model.height))
    boxes = do_detect(model, sized, 0.5, 0.4, use_cuda)

    width = img.width
    height = img.height
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height
        box[0] = x1
        box[1] = y1
        box[2] = x2
        box[3] = y2
    return boxes


def detect(cfgfile, weightfile, images, use_cuda=0):
    cats = ["data/birds1.jpg", "data/birds2.jpg"]
    embs = []

    for cat in cats:
        m = Darknet(cfgfile)

        m.print_network()
        m.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))

        if m.num_classes == 20:
            namesfile = 'data/voc.names'
        elif m.num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'

        if use_cuda:
            m.cuda()

        img = Image.open(cat).convert('RGB')
        sized = img.resize((m.width, m.height))

        for i in range(1):
            start = time.time()
            # boxes, embed = do_detect(m, sized, 0.5, 0.4, use_cuda)
            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

        class_names = load_class_names(namesfile)
        plot_boxes(img, boxes, 'predictions.jpg', class_names)

        temp = []
        for box in boxes:
            temp.append(box[-1].view(1, -1))
        embs.append(temp)
    dist = torch.nn.modules.PairwiseDistance()
    print("=====================")
    for i in range(len(embs)):
        for j in range(len(embs[i])):
            hero = embs[i][j]
            for frame in range(len(embs)):
                for box in range(len(embs[frame])):
                    if box == j:
                        print(i, j, frame, box, "Should be small:",
                              dist(hero.view(1, -1), embs[frame][box].view(1, -1)))
                    else:
                        print(i, j, frame, box, "Should be big:", dist(hero.view(1, -1), embs[frame][box].view(1, -1)))

    print("=====================")


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 0
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(1):
        start = time.time()
        # boxes, embed = do_detect(m, sized, 0.5, 0.4, use_cuda)
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(1):
        start = time.time()
        # boxes, embed = do_detect(m, sized, 0.5, 0.4, use_cuda)
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 0:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)
    print()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        detect(cfgfile, weightfile, imgfile)
        # detect_cv2(cfgfile, weightfile, imgfile)
        # detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')

        detect('cfg/yolo.cfg', 'yolo.weights', 'data/dog.jpg')
