

from PIL import Image
import random





def get_random_crop(image: Image, height: int, width: int)->Image:
    x, y = image.size
    x1 = random.randrange(0, max(1, x - height))
    y1 = random.randrange(0,  max(1, y - width))
    return image.crop((x1, y1, x1 + height, y1 + width))

def get_random_flip(image: Image)->Image:
    flip = random.choice([True, False])
    if flip:
        image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    return image



def get_random_augment(image: Image, crop_size: tuple, crop=True, flip=True)->Image:
    if crop and len(crop_size) >= 2:
        image = get_random_crop(image, crop_size[0], crop_size[1])
    if flip:
        image = get_random_flip(image)
    return image



