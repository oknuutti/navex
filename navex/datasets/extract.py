import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tr

from .base import DataLoadingException, IdentityTransform


class SingleImageDataset(Dataset):
    def __init__(self, path, eval=True, rgb=False):
        if eval:
            self.transforms = tr.Compose([
                tr.Grayscale(num_output_channels=1),
                tr.ToTensor(),
                tr.Normalize(mean=[0.449], std=[0.226]),
            ])
        else:
            assert False, 'not implemented'

        if rgb:
            self.transforms.transforms[0] = IdentityTransform()

        # load samples
        if os.path.isdir(path):
            self.samples = [os.path.join(path, f.strip()) for f in os.listdir(path)
                                                  if f[-4:] in ('.jpg', '.png', '.bmp', '.jpeg', '.ppm')]
        else:
            with open(path) as fh:
                path = os.path.dirname(path)
                self.samples = list(map(lambda x: os.path.join(path, x.strip()), fh))

    def __getitem__(self, idx):
        try:
            img = Image.open(self.samples[idx])
            if self.transforms is not None:
                img = self.transforms(img)
        except DataLoadingException as e:
            raise DataLoadingException("Problem with idx %s:\n%s" % (idx, self.samples[idx],)) from e

        return img

    def __len__(self):
        return len(self.samples)
