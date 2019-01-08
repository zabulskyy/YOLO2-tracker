import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import cv2

"""==========================Oles's code==========================="""


def draw_rectangles(img, boxes):
    for i in range(len(boxes)):
        box = boxes[i]
        rgb = (255, 0, 0)
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), rgb, 1)
    return img


# def vizualize_yolo_results("bag"):
#     path = "/home/zabulskyy/YOLO2-tracker/notebooks/heatmap_results/{}.csv".format(name)


def draw_centers(img, boxes):
    for i in range(len(boxes)):
        box = boxes[i]
        rgb = (255, 0, 0)
        # print(box)
        center = (int((box[1] + box[3]) / 2), int((box[2] + box[4]) / 2))
        img = cv2.circle(img, center, 5, rgb, -1)
        img = cv2.putText(img, str(i), center, 1, 2, rgb)
    return img


def analyze_object(name):
    path = "/home/zabulskyy/YOLO2-tracker/notebooks/heatmap_results/{}.csv".format(name)
    df = pd.read_csv(path, header=None)
    df.columns = pd.read_csv("/home/zabulskyy/YOLO2-tracker/notebooks/header.txt", header=None).values[0]
    # print(df.columns)
    frames_ids = df["frame_number"].unique()
    # print(df.head())
    # old_embedding = None
    # new_embedding = None
    #
    # objects = {}
    tracks = []
    k = 10
    frame_path = "/data/vot2016/" + name + "/00000001.jpg"
    img = cv2.imread(frame_path)
    out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, img.shape[1::-1])
    for frame_id in frames_ids:
        objects = df[df["frame_number"] == frame_id].values
        # matched = [False] * len(objects)
        for i, object in enumerate(objects):

            # if matched[i]:
            #     continue

            object = objects[i]
            grid_position = np.array(object[-2:], int)
            embedding = object[8:1032]
            box = object[1:5].tolist()

            best_track = None
            best_track_dist = .3
            best_track_num = None

            for n, track in enumerate(tracks):
                emb_dist = np.mean((track["embedding"][-1] - embedding) ** 2)
                if emb_dist < best_track_dist:
                    best_track_dist = emb_dist
                    best_track = track
                    best_track_num = n

            if best_track is not None:
                # matched[i] = True
                best_track["grid_position"].append(grid_position)
                best_track["embedding"].append(embedding)
                best_track["box"].append(box)
                best_track["is_used"] = True
                best_track["frame_id"].append(frame_id)

            # if not track["is_used"] and \
            #         np.mean((track["grid_position"][-1] - grid_position) ** 2) <= 1 and \
            #         np.mean((track["embedding"][-1] - embedding) ** 2) <= 0.3:

        if frame_id < k:
            for i in range(len(objects)):
                if True:  # not matched[i]:
                    object = objects[i]
                    grid_position = np.array(object[-2:], int)
                    embedding = object[8:1032]
                    box = object[1:5].tolist()
                    tracks.append(
                        {"frame_id": [frame_id], "grid_position": [grid_position], "embedding": [embedding],
                         "box": [box], "is_used": True})

        boxes = []
        for i, track in enumerate(tracks):
            if track["is_used"]:
                box = track["box"][-1].copy()
                box.insert(0, i)
                boxes.append(box)

        frame_path = "/data/vot2016/" + name + "/" + "0" * (8 - len(str(frame_id + 1))) + str(frame_id + 1) + ".jpg"
        img = cv2.imread(frame_path)
        img = draw_centers(img, boxes)
        # print(frame_id)
        out.write(img)

        for track in tracks:
            track["is_used"] = False

        # print("=={}==".format(frame_id))
        # # print(tracks)
        # for track in tracks:
        #     print(track["grid_position"][-1], len(track["grid_position"]))
    # for track in tracks:
    #     print(track["embedding"])
    #     print(track["grid_position"])
    out.release()
    return tracks, df
    # print(objects[:,-2:])

    ##.values[0, 8:1032]
    ##objects[frame_id].append()
    #
    # if old_embedding is None:
    #     old_embedding = embedding
    # else:
    #     # new_embedding, old_embedding = embedding, new_embedding
    #     print(np.mean((embedding - old_embedding) ** 2))
    # print(len(data))
    # print(data[:20])
    # break
    # cv2.imread("data/vot2016/bag/00000001.jpg")


""""=====================Volodymyr's code======================"""


def iou(box_a, box_b):
    # IoU between two boxes
    box_a = box_a[1:5] if len(box_a) != 4 else box_a
    box_b = box_b[1:5] if len(box_b) != 4 else box_b

    x_a = np.max((box_a[0], box_b[0]))
    y_a = np.max((box_a[1], box_b[1]))
    x_b = np.min((box_a[2], box_b[2]))
    y_b = np.min((box_a[3], box_b[3]))

    inter_area = np.max((0., x_b - x_a + 1)) * np.max((0., y_b - y_a + 1))
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    a = (box_a_area + box_b_area - inter_area)
    res = inter_area / a
    return res


def best_iou(gt, candidates):
    best_iou = 0
    best_index = 0
    _iou = 0
    for i, candidate in enumerate(candidates):
        _iou = iou(candidate, gt)
        if _iou > best_iou:
            best_iou = _iou
            best_index = i
    return best_index if _iou > 0 else None


def force_square(m):
    return [min(m[::2]), min(m[1::2]), max(m[::2]), max(m[1::2])]


def get_emb_by_frame_id(track, frame_id):
    pass


def compare_embeddings(class_name, df, data_path="/data/vot2016", results=None, verbose=False):
    if results is None:
        results = list()
    images = sorted([x for x in os.listdir(osp.join(data_path, class_name)) if x.endswith("jpg")])
    with open(osp.join(data_path, class_name, "groundtruth.txt"), "r") as file:
        f = file.read().split()
        gts = torch.tensor([[i] + force_square(list(map(float, f[i].split(',')))) for i in range(len(f))]).double()

    hero_embs = []
    rest_embs = []
    frames_ids = df["frame_number"].unique()
    prev_hero = None

    for frame_id in frames_ids:
        objects = np.array(df[df["frame_number"] == frame_id].values)
        _best_iou = best_iou(gts[frame_id], objects)
        if _best_iou is None:
            continue
        hero = objects[_best_iou]
        if prev_hero is None:
            prev_hero = hero
            continue
        hero_emb_diff = np.mean((hero[8:1032] - prev_hero[8:1032]) ** 2)
        if verbose:
            print("--- emb diff with the gt: ---")
            print(hero_emb_diff)
            print("--- other embeddings: ---")
        other_embs_diff = []
        for object in objects:
            e = np.mean((object[8:1032] - prev_hero[8:1032]) ** 2)
            if verbose:
                print(e)
            other_embs_diff.append(e)
        hypothesis_true = np.min(other_embs_diff) - .001 <= hero_emb_diff <= np.min(other_embs_diff) + .001
        if verbose:
            print("--- hypothesis true? ---")
            print(hypothesis_true)
        prev_hero = hero
        results.append(int(hypothesis_true))
    return results


# print(df.head())
# for i in range(13):
#     for j in range(13):
#        #
#
#         print(i, j)

if __name__ == "__main__":
    # with open(r"accuracy_results2.csv", 'r') as f:
    #     lines = [float(x.split(",")[1]) for x in f.read().split()]
    #     print("mean:{}, median:{}".format(np.mean(lines), np.median(lines)))
    #
    # exit(100500)
    names = sorted([x for x in os.listdir("/data/vot2016") if not x.endswith("txt")])
    for name in ["matrix"]:  # names:
        print(name)
        print(name, file=open(r"accuracy_results2.csv", "a+"), end=",")
        tracks, df = analyze_object(name)
        results = compare_embeddings(name, df)
        print(np.mean(results))
        print(np.mean(results), file=open(r"accuracy_results2.csv", "a+"))

    # frame_id = 200
    #
    # path = "/home/zabulskyy/YOLO2-tracker/notebooks/heatmap_results/{}.csv".format(name)
    # df = pd.read_csv(path, header=None)
    # df.columns = pd.read_csv("/home/zabulskyy/YOLO2-tracker/notebooks/header.txt", header=None).values[0]
    # print(df.columns)
    # frame_path = "/data/vot2016/" + name + "/" + "0" * (8 - len(str(frame_id))) + str(frame_id) + ".jpg"
    # print(frame_path)
    # img = cv2.imread(frame_path)
    # boxes = df[df["frame_number"] == frame_id].values[:,1:]
    # print(boxes, len(boxes))
    # img = draw_rectangles(img, boxes)
    # plt.imshow(img)
    # plt.show()
