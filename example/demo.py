import selectivesearch
from selectivesearch.helpers import visualize_regions
import numpy as np
from PIL import Image
a= Image.open('./2016-sand-cat-group.jpg', 'r')
rgb_im = a.convert('RGB')
img = np.array(rgb_im)
img = img.astype('float32') / 255
print(img.shape)
rects = set(selectivesearch.selectivesearch(img, min_size=300, threshold=1000))
visualize_regions(img, rects)
