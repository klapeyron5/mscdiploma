import json
from . import mappings, video_example, video
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_alignment
from copy import deepcopy
from .ms_facereconstruction.main import MS
import tensorflow as tf

def main():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    ms = MS()
    def get_lms5_2d_from_2dFan(img):
        if any([x<32 for x in img.shape[:2]]):
            return None
        preds = fa.get_landmarks(img)
        if preds is None:
            return None
        Lm2D = preds[0]
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1

        l_eye = np.mean(Lm2D[lm_idx[[1, 2]]], 0)
        r_eye = np.mean(Lm2D[lm_idx[[3, 4]]], 0)
        nose = Lm2D[lm_idx[0]]
        l_mouth = Lm2D[lm_idx[5]]
        r_mouth = Lm2D[lm_idx[6]]

        lms5 = np.stack([l_eye, r_eye, nose, l_mouth, r_mouth], axis=0)
        lms5 = np.round(lms5).astype(int)
        return lms5

    def draw_vertices(mesh_vertices, image, vertice_color=[255, 255, 255], vertice_r=1, extra_scale=3):
        mesh_vertices *= extra_scale

        h, w, _ = image.shape
        w *= extra_scale
        h *= extra_scale
        image = cv2.resize(image, (w, h))

        def draw_r_1():
            image[y, x] = vertice_color
        def draw_r():
            cv2.circle(image, (x, y), vertice_r, vertice_color, thickness=-1)

        if vertice_r == 1:
            draw_func = draw_r_1
        else:
            draw_func = draw_r

        for vertex in mesh_vertices:
            x, y = [np.round(v).astype(int) for v in vertex[:2]]
            if 0 <= x < w and 0 <= y < h:
                draw_func()
        return image


    mp = mappings.GetWorkingDir()
    file = os.path.join(mp(**{})[mp.PIPE_DEFAULT.KEY_OUT_working_dir], 'SVO-03-01-detect.json')
    assert os.path.isfile(file)
    data = json.load(open(file))
    frame_img_file = os.path.join(mp.PIPE_DEFAULT.KEY_OUT_working_dir, 'tmp_sb/0001.jpg')
    assert os.path.isfile(frame_img_file)
    frame_img = cv2.imread(frame_img_file)
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)


    vis_scale = 4
    h, w, _ = frame_img.shape
    w *= vis_scale
    h *= vis_scale
    frame_img_vis = cv2.resize(frame_img, (w, h))


    keypoints_ref = dict(
        nose = 0,
        leye = 15,
        reye = 16,
        lear = 17,
        rear = 18,
        chest = 1,
        lshoulder = 2,
        rshoulder = 5,
    )

    for frame in data['FrameSequences'][0]['Frames']:
        for detected_object in frame['DetectedObjects'][0:]:
            if 'Bones' in detected_object:
                bones = detected_object['Bones']
                points = []
                for x,y in zip(bones[::2], bones[1::2]):
                    if x==-1 and y==-1:
                        pass
                    else:
                        assert x>=0 and y>=0, "{}, {}".format(x,y)
                    points.append([x, y])

                points = np.array(points)
                points = points[list(keypoints_ref.values())]
                points = np.array(list(filter(lambda p: (p[0]>=0)and(p[1]>=0), points)))
                if len(points)==0: continue

                stds = 3*np.array([np.std(points[:, 0]), 1.5*np.std(points[:, 1])])
                if not 0.1<stds[0]/stds[1]<10: continue
                center = np.array(np.mean(points, axis=0))
                assert center.shape==stds.shape
                bbox = [center[0]-stds[0], center[1]-stds[1], center[0]+stds[0], center[1]+stds[1]]
                bbox = np.round(bbox).astype(int)

                #
                # try:
                #     assert all([x>=0 for x in nose])
                #     assert all([x>=0 for x in chest])
                # except Exception:
                #     continue

                # bbox_h = abs(nose[1]-chest[1])/2
                # frame_img = cv2.rectangle(frame_img, (int(round(nose[0]-bbox_h)), int(round(nose[1]-bbox_h))),
                #               (int(round(nose[0]+bbox_h)), int(round(nose[1]+bbox_h))),
                #               color=(0,255,255), thickness=2)

                # frame_img = cv2.rectangle(frame_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,255,255), thickness=2)
                crop_from_openpose = frame_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                # plt.imshow(frame_img)
                # plt.show()

                # plt.imshow(crop_from_openpose)
                # plt.title('crop_from_openpose')
                # plt.show()
                lms2d_n5 = get_lms5_2d_from_2dFan(deepcopy(crop_from_openpose))
                if lms2d_n5 is None:
                    continue

                vis_crop = deepcopy(crop_from_openpose)
                for p in lms2d_n5:
                    vis_crop = cv2.circle(vis_crop, tuple([int(x) for x in p]), radius=1, color=(0,255,0), thickness=-1)
                # plt.imshow(vis_crop)
                # plt.title('crop_from_openpose with lms2d_n5')
                # plt.show()

                img = np.expand_dims(deepcopy(crop_from_openpose), axis=0)
                lms2d_n5 = np.expand_dims(lms2d_n5, axis=0)
                out = ms.get_depthmap(img, lms2d_n5, False)
                if any([x==0 for x in out['crop'].numpy().shape]):
                    continue
                plt.imshow(out['crop'].numpy().astype(int))
                plt.title('ms_crop')
                plt.show()
                mesh = out['mesh'].numpy()[0]
                # vis = draw_vertices(mesh, out['crop'].numpy())
                # vis = vis.astype(int)
                # for p in lms2d_n5:
                #     vis = cv2.circle(vis, tuple([int(x) for x in p]), radius=1, color=(0,255,0), thickness=-1)
                # plt.imshow(vis)
                # plt.title('ms_crop')
                # plt.show()
                # print()



                t = out['t'].numpy()
                t[0] *= crop_from_openpose.shape[1]
                t[1] *= crop_from_openpose.shape[0]
                # t = np.round(t).astype(int)
                mesh = out['mesh'].numpy()[0]
                mesh *= 1/out['s'].numpy()
                mesh[:,0]+=t[0]
                mesh[:,1]+=t[1]
                # vis = draw_vertices(deepcopy(mesh), deepcopy(crop_from_openpose), extra_scale=6)
                # vis = vis.astype(int)
                # plt.imshow(vis)
                # plt.title('openpose_crop')
                # plt.show()
                # print()


                # ms_crop = np.round(out['crop'].numpy()).astype(int)
                # new_shape = np.round(1/out['s'].numpy()*np.array(ms_crop.shape[:2])).astype(int)
                # a = np.round(tf.image.resize(ms_crop, new_shape[:2],).numpy()).astype(int)
                # t = np.round(t).astype(int)
                # crop_from_openpose[t[1]:t[1]+new_shape[0], t[0]:t[0]+new_shape[1], :] = a
                # plt.imshow(crop_from_openpose)
                # plt.title('ms crop insertion')
                # plt.show()
                # print()

                mesh[:, 0] += bbox[0]
                mesh[:, 1] += bbox[1]
                mesh *= vis_scale
                frame_img_vis = draw_vertices(deepcopy(mesh), deepcopy(frame_img_vis), extra_scale=1)
                frame_img_vis = frame_img_vis.astype(int)
                plt.imshow(frame_img_vis)
                plt.title('main frame')
                plt.show()
                print()
            # break
        break
    print()
