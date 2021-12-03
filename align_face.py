import os
from tqdm import *
from skimage import io
from shutil import copyfile
import cv2
import numpy as np 
import imgaug as ia
from imgaug import augmenters as iaa
from skimage import transform as trans
from shutil import copyfile
import face_alignment
import argparse

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type = str, help='path to your dataset')
parser.add_argument('--dst_w', type = int, default = 112, help='output image width')
parser.add_argument('--dst_h', type = int, default = 112, help='output image height')
parser.add_argument('--resize', type = bool, default = False, help='resize image before aligning')
args = parser.parse_args()

def alignment(cv_img, dst, dst_w, dst_h):
    if dst_w == 96 and dst_h == 112:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32)
    elif dst_w == 112 and dst_h == 112:
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32)
    elif dst_w == 150 and dst_h == 150:
        src = np.array([
            [51.287415, 69.23612],
            [98.48009, 68.97509],
            [75.03375, 96.075806],
            [55.646385, 123.7038],
            [94.72754, 123.48763]], dtype=np.float32)
    elif dst_w == 160 and dst_h == 160:
        src = np.array([
            [54.706573, 73.85186],
            [105.045425, 73.573425],
            [80.036, 102.48086],
            [59.356144, 131.95071],
            [101.04271, 131.72014]], dtype=np.float32)
    elif dst_w == 224 and dst_h == 224:
        src = np.array([
            [76.589195, 103.3926],
            [147.0636, 103.0028],
            [112.0504, 143.4732],
            [83.098595, 184.731],
            [141.4598, 184.4082]], dtype=np.float32)
    else:
        return None
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    face_img = cv2.warpAffine(cv_img,M,(dst_w,dst_h), borderValue = 0.0)
    return face_img

if __name__ == '__main__':
    root_dir = args.root_dir
    for folder in os.listdir(root_dir):
        print("Process id: ",folder)
        if not os.path.exists('./data/aligned_'+str(args.dst_w)+str(args.dst_h)+'/'+folder):
            os.makedirs('./data/aligned_'+str(args.dst_w)+str(args.dst_h)+'/'+folder)

        for file in tqdm(os.listdir(root_dir+'/'+folder)):
            img_path = root_dir+'/'+folder+'/'+file
            image = io.imread(img_path)
            for sigma in np.linspace(0.0, 3.0, num=11).tolist():
                seq = iaa.GaussianBlur(sigma)
                image_aug = seq.augment_image(image)
                landmarks = fa.get_landmarks(image_aug)
                if landmarks is not None:
                    # print('sigma:',sigma)
                    points = landmarks[0]
                    p1 = np.mean(points[36:42,:], axis=0)
                    p2 = np.mean(points[42:48,:], axis=0)
                    p3 = points[33,:]
                    p4 = points[48,:]
                    p5 = points[54,:]

                    if np.mean([p1[1],p2[1]]) < p3[1] \
                        and p3[1] < np.mean([p4[1],p5[1]]) \
                        and np.min([p4[1], p5[1]]) > np.max([p1[1], p2[1]]) \
                        and np.min([p1[1], p2[1]]) < p3[1] \
                        and p3[1] < np.max([p4[1], p5[1]]):

                        dst = np.array([p1,p2,p3,p4,p5],dtype=np.float32)
                        cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        if args.resize:
                            cv_img=cv2.resize(cv_img, [args.dst_w, args.dst_h])                     

                        output_face = alignment(cv_img, dst, args.dst_w, args.dst_h)
                save_path = './data/aligned_'+str(args.dst_w)+str(args.dst_h)+'/'+folder+'/'+file
                cv2.imwrite(save_path, output_face)

