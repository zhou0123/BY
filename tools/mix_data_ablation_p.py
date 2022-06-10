import json
import os


"""
cd datasets
mkdir -p mix_mot_ch/annotations
cp mot/annotations/val_half.json mix_mot_ch/annotations/val_half.json
cp mot/annotations/test.json mix_mot_ch/annotations/test.json
cd mix_mot_ch
ln -s ../mot/train mot_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
ln -s ../crowdhuman/CrowdHuman_val crowdhuman_val
cd ..
"""

mot_json = json.load(open('datasets/mot/annotations/train_half.json','r'))

img_list = list()
ann_list = list()

for img in mot_json['images']:
    img['file_name'] = 'train17/' + img['file_name']
    img_list.append(img)
for ann in mot_json['annotations']:
    ann_list.append(ann)
video_list = mot_json['videos']
category_list = mot_json['categories']
print('mot17')
mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('datasets/mix_mot_ch/annotations/train17.json','w'))

img_list = list()
ann_list = list()
mot_json15 = json.load(open('datasets/MOT15/annotations/train.json','r'))
for img in mot_json15['images']:
    img['file_name'] = 'train15/' + img['file_name']
    img_list.append(img)
for ann in mot_json15['annotations']:
    ann_list.append(ann)
video_list = video_list+mot_json15['videos']
category_list = category_list+mot_json15['categories']
print('mot15')
mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('datasets/mix_mot_ch/annotations/train15.json','w'))


img_list = list()
ann_list = list()
mot_json16 = json.load(open('datasets/MOT16/annotations/train.json','r'))
for img in mot_json16['images']:
    img['file_name'] = 'train16/' + img['file_name']
    img_list.append(img)
for ann in mot_json16['annotations']:
    ann_list.append(ann)
video_list = video_list+mot_json16['videos']
category_list = category_list+mot_json16['categories']
print('mot16')
mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('datasets/mix_mot_ch/annotations/train16.json','w'))


img_list = list()
ann_list = list()
mot_json20 = json.load(open('datasets/MOT20/annotations/train.json','r'))
for img in mot_json20['images']:
    img['file_name'] = 'train20/' + img['file_name']
    img_list.append(img)
for ann in mot_json20['annotations']:
    ann_list.append(ann)

video_list = video_list+mot_json20['videos']
category_list = category_list+mot_json20['categories']
print('mot20')
mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('datasets/mix_mot_ch/annotations/train20.json','w'))