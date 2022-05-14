import json
import os
import random

def split_dataset(root, input_json, ratio, seed):
    random.seed(seed)

    input_json = os.path.join(root, input_json)
    with open(input_json) as f:
        dataset = json.load(f)
        
    imgs = dataset['images']
    anns = dataset['annotations']
    
    image_ids = [x['id'] for x in imgs]
    random.shuffle(image_ids)

    n_train = int(len(image_ids) * (1-ratio))
    train_ids = set(image_ids[:n_train])
    train_imgs = [x for x in imgs if x['id'] in train_ids]
    train_anns = [x for x in anns if x['image_id'] in train_ids]
    
    val_ids = set(image_ids[n_train:])
    val_imgs = [x for x in imgs if x['id'] in val_ids]
    val_anns = [x for x in anns if x['image_id'] in val_ids]

    train_data = {
        'images': train_imgs,
        'annotations': train_anns,
        'categories': dataset['categories'],
    }

    val_data = {
        'images': val_imgs,
        'annotations': val_anns,
        'categories': dataset['categories'],
    }

    output_json_train = os.path.join(root, f'train_{seed}.json')
    output_json_val = os.path.join(root, f'val_{seed}.json')

    with open(output_json_train, 'w') as f:
        json.dump(train_data, f)

    with open(output_json_val, 'w') as f:
        json.dump(val_data, f)
        
        