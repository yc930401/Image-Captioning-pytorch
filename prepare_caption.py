import json

train_caption_dir = 'F:/Dataset for Deep Learning/Image/Image Captioning-Coco2014/annotations/captions_train2014.json'
val_caption_dir = 'F:/Dataset for Deep Learning/Image/Image Captioning-Coco2014/annotations/captions_val2014.json'
output_train_dir = 'data/captions_train2014.txt'
output_val_dir = 'data/captions_val2014.txt'

input_dirs = [train_caption_dir, val_caption_dir]
output_dirs = [output_train_dir, output_val_dir]
for i in range(2):
    input_dir = input_dirs[i]
    output_dir = output_dirs[i]
    # read
    with open(input_dir, 'r') as f1:
        json_data = json.load(f1)
        annptations = json_data['annotations']
        id_to_captions = {item['image_id']: item['caption'] for item in annptations}
        images = json_data['images']
        id_to_images = {item['id']: item['file_name'] for item in images}
        image_to_caption = {id_to_images[id]: caption for id, caption in id_to_captions.items()}
    # write
    with open(output_dir, 'w') as f2:
        for image, caption in image_to_caption.items():
            f2.write(image + '\t' + caption + '\n')
