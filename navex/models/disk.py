
# install by `pip install git+https://github.com/jatentaki/unets`,
#            `pip install git+https://github.com/jatentaki/torch-localize` and
#            `pip install git+https://github.com/jatentaki/torch-dimcheck`
import unets


from .r2d2 import R2D2


class DISK(R2D2):
    def __init__(self, arch, des_head, det_head, qlt_head, **kwargs):
        super(DISK, self).__init__(arch, des_head, det_head, qlt_head, **kwargs)
        if self.separate_des_head:
            self.des_head = None
            self.det_head = None
            self.qlt_head = None

    def create_backbone(self, arch, cache_dir=None, pretrained=False, width_mult=1.0, in_channels=1,
                        separate_des_head=False, **kwargs):
        bb_ch_out = self.conf['descriptor_dim'] \
                    + ((1 + (0 if self.conf['qlt_head']['skip'] else 1)) if separate_des_head else 0)

        kernel_size = 5
        down_channels = [16, 32, 64, 64, 64]
        up_channels = [64, 64, 64, bb_ch_out]
        setup = {**(unets.fat_setup if arch == 'fat' else unets.thin_setup), 'bias': True, 'padding': True}

        unet = unets.Unet(in_features=in_channels, size=kernel_size, down=down_channels, up=up_channels, setup=setup)

        return unet, up_channels[-1]

    def forward(self, input):
        # input is a pair of images
        features = self.backbone(input)

        if self.separate_des_head:
            dn = self.conf['descriptor_dim']
            des = features[:, :dn, :, :]
            det = features[:, dn: dn+1, :, :]
            qlt = None if self.conf['qlt_head']['skip'] else features[:, -1:, :, :]
        else:
            des = features if getattr(self, 'des_head', None) is None else self.des_head(features)
            des2 = None

            if self.conf['det_head']['after_des']:
                des2 = des.pow(2) if des2 is None else des2
                det = self.det_head(des2)
            else:
                det = self.det_head(features)

            if self.qlt_head is None:
                qlt = None
            elif self.conf['qlt_head']['after_des']:
                des2 = des.pow(2) if des2 is None else des2
                qlt = self.qlt_head(des2)
            else:
                qlt = self.qlt_head(features)

        output = self.fix_output(des, det, qlt)
        return output
