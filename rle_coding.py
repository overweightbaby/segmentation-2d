import numpy as np
from skimage.measure import label
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from myUnet import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import csv

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right  返回条件为真的坐标的起始位置
    #print(dots)
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x,cutoff=0.5):
    lab_img = label(x>cutoff,connectivity=2)
    print(lab_img.max())
    for i in range(1):  #连通区域块的数目 从0开始标记的,lab_img.max()+1
        return rle_encoding(lab_img==1)


def get_rle():
    weights_path = 'model/unet-0.8239.hdf5'
    imgs_path = '/data2/zyj/kaggle/test/images'
    imgs = os.listdir(imgs_path)
    rows = []
    test = []
    names = []
    for img in imgs:
        img_path = os.path.join(imgs_path,img)
        print(img)
        img_ = load_img(img_path)
        img_array = img_to_array(img_)
        print(img_array.shape)
        names.append(img)
        test.append(img_array[0:96,0:96,:])
        test.append(img_array[-97:-1,-97:-1,:])
    test = np.asarray(test)
    print(test.shape)
    myunet = myUnet(bn_learn=False)
    model = myunet.get_unet()
    model.load_weights(weights_path)
    score = model.predict(test)
    score = myunet.fill_hole(score)
    final_pred = np.zeros((int(test.shape[0]/2),101,101,3))
    for i in range(int(test.shape[0]/2)):
        final_pred[i,0:96,0:96,:]=score[i*2,:,:,:]
        final_pred[i,96::,96::,:]=score[2*i+1,-6:-1,-6:-1,:]
        rle_code = prob_to_rles(final_pred[i])
        row=[names[i],rle_code]
        rows.append(row)
        print(i)
        with open('test.csv','w') as csvfile:
            writer = csv.writer(csvfile,delimiter=',') 
            row=['id','rle_mask']
            for row1 in rows:
                writer.writerow(row1)


if __name__ == '__main__':
    get_rle()
