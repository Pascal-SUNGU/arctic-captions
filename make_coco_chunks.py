from caffe_cnn import *
import pandas as pd
import numpy as np
import os
import scipy
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
import pdb
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import requests

#Create the CNN

vgg_deploy_path = 'VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_model_path  = 'VGG_ILSVRC_19_layers.caffemodel'
cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=20,
          width=224,
          height=224)



#Get filenames for training/testing
coco_image_path = '/usr0/multicomp/datasets/coco/images'
tpath = '/usr0/multicomp/datasets/coco/images/train2014/'
vpath = '/usr0/multicomp/datasets/coco/images/val2014/'

#Get train data from the training file
t_annFile = '/usr0/multicomp/datasets/coco/annotations/captions_train2014.json'
v_annFile = '/usr0/multicomp/datasets/coco/annotations/captions_val2014.json'

with open('./splits/coco_train.txt','r') as f:
    trainids = [x for x in f.read().splitlines()]
with open('./splits/coco_restval.txt','r') as f:
    trainids += [x for x in f.read().splitlines()]
with open('./splits/coco_val.txt','r') as f:
    valids = [x for x in f.read().splitlines()]
with open('./splits/coco_test.txt','r') as f:
    testids = [x for x in f.read().splitlines()]

#Another fast representation: by dictionary
whatType = {}
for t in trainids:
    whatType[t] = "train"
for t in valids:
    whatType[t] = "val"
for t in testids:
    whatType[t] = "test"


#Extract from json
val = json.load(open(v_annFile, 'r'))
train = json.load(open(t_annFile, 'r'))
imgs = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

trainImgs = []
valImgs = []
testImgs = []

train_id2idx = {}
val_id2idx = {}
test_id2idx = {}
trainidx = 0
validx = 0
testidx = 0

for img in imgs:
    thetype = whatType[img['file_name']]
    if thetype == "train":
        trainImgs.append(img)
        train_id2idx[img['id']] = trainidx
        trainidx += 1
    elif thetype == "val":
        valImgs.append(img)
        val_id2idx[img['id']] = validx
        validx += 1
    elif thetype == "test":
        testImgs.append(img)
        #print img.keys()
        test_id2idx[img['id']] = testidx
        testidx += 1


# for efficiency lets group annotations by image file ID
itoa = {}
allAnnots = []
for a in annots:
    imgid = a['id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)
    allAnnots.append(a['caption'])


#annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
#annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
#annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))
#================
#Vectorize things
#================
# vectorizer = CountVectorizer(encoding='unicode',lowercase=False).fit(allAnnots)
# dictionary = vectorizer.vocabulary_
# dictionary_series = pd.Series(dictionary.values(), index=dictionary.keys()) + 2
# dictionary = dictionary_series.to_dict()


# # Sort dictionary in descending order
# #TODO: Sort by frequency
# from collections import OrderedDict
# dictionary = OrderedDict(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))

# with open('./data/coco/dictionary.pkl', 'wb') as f:
#     cPickle.dump(dictionary, f)

####Where I'm at now

def makeCaps(imgList,ind_dict):
    newCaps = []
    for timg in imgList:
        theid = ind_dict[timg['id']]
        newCaps.append((allAnnots[theid],theid))
    return newCaps

cap_train = makeCaps(trainImgs,train_id2idx)
cap_val = makeCaps(valImgs,val_id2idx)
cap_test = makeCaps(testImgs,test_id2idx)
print "done with linking"
#Gotta do this now

def getFilename(imgobj):
    fn = imgobj['file_name']
    if fn.startswith('COCO_val'):
        return vpath + fn
    return tpath + fn


def processImgList(theList,basefn):
    batch_size = 100
    numPics = 0
    batchNum = 0

    for start, end in zip(range(0, len(theList)+100, 100), range(100, len(theList)+100, 100)):
        print("processing images %d to %d" % (start, end))
        image_files = [getFilename(x) for x in theList[start:end]]
        feat = cnn.get_features(image_list=image_files, layers='conv5_4', layer_sizes=[512,14,14])
        if numPics % batch_size == 0: #reset!
            featStacks = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
        else:
            featStacks = scipy.sparse.vstack([featStacks, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))],format="csr")
        
        numPics += 1

        if numPics % batch_size == 0:
            newfn = basefn + str(batchNum) + '.pkl'
            with open(newfn,'wb') as f:
                cPickle.dump(featStacks, f,protocol=cPickle.HIGHEST_PROTOCOL)
                print("Success!")
            batchNum += 1

    if numPics % batch_size != 0:
        newfn = basefn + str(batchNum) + '.pkl'
        with open(newfn,'wb') as f:
            cPickle.dump(featStacks, f,protocol=cPickle.HIGHEST_PROTOCOL)
    return featStacks


print('train now')
train_feats = processImgList(trainImgs,'./data/coco_align.train')
# dumpStuff('./data/coco_align.train.npy',train_feats)
with open('./data/coco_align.train.pkl', 'wb') as f:
    # cPickle.dump(train_feats, f,protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(cap_train, f,protocol=cPickle.HIGHEST_PROTOCOL)

print('val now')
val_feats = processImgList(valImgs,'./data/coco_align.val')
# dumpStuff('./data/coco_align.val.npy',val_feats)
with open('./data/coco_align.val.pkl', 'wb') as f:
    # cPickle.dump(val_feats, f,protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(cap_val, f,protocol=cPickle.HIGHEST_PROTOCOL)

print('test now')
test_feats = processImgList(testImgs,'./data/coco_align.test')
# dumpStuff('./data/coco_align.test.npy',test_feats)
with open('./data/coco_align.test.pkl', 'wb') as f:
    # cPickle.dump(test_feats, f,protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(cap_test, f,protocol=cPickle.HIGHEST_PROTOCOL)


yoparams = {'api_token':'4deecd88-ae42-8e50-4ce6-14d4c5c59873'}
yoall = requests.post('http://api.justyo.co/yoall/',data=yoparams)