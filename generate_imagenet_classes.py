# generate_imagenet_classes.py

import json

# Download from: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
import requests

response = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
class_idx = json.loads(response.content)

with open('imagenet_classes.txt', 'w') as f:
    for idx in range(len(class_idx)):
        f.write(f"{class_idx[str(idx)][1]}\n")
