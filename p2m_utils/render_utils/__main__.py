import argparse
from p2m_utils.render_utils import ObjRenderer, PCRenderer
import numpy as np
from skimage.morphology import binary_closing
from PIL import Image

renderers = {
    "obj": ObjRenderer,
    "pc": PCRenderer
}

kernel = np.ones((5, 5))

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("filetype", choices=list(renderers.keys()))
args = parser.parse_args()

renderer = renderers[args.filetype]()
image = renderer.render(args.filename, "front")
npim = image
filled = np.where(binary_closing(npim, kernel), 255, 0).astype(np.uint8)
Image.fromarray(filled).save("out-front.png".format(args.filename))


image = Image.fromarray(np.where(binary_closing(renderer.render(args.filename, "side"), kernel), 255, 0).astype(np.uint8))
image.save("out-side.png".format(args.filename))
