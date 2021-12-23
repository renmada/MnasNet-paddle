import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class OctConv(nn.Layer):
    def __init__(self, out_channels, kernel_size, strides=(1, 1), use_bias=True,
                 in_channels=0, enable_path=((0, 0), (0, 0)), padding=0,
                 groups=1):
        super().__init__()
        (h2l, h2h), (l2l, l2h) = enable_path
        c_h, c_l = out_channels if type(out_channels) is tuple else (out_channels, 0)
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        is_dw = False
        # computational graph will be automatic or manually defined
        self.enable_l2l = True if l2l != -1 and (in_c_l >= 0 and c_l > 0) else False
        self.enable_l2h = True if l2h != -1 and (in_c_l >= 0 and c_h > 0) else False
        self.enable_h2l = True if h2l != -1 and (in_c_h >= 0 and c_l > 0) else False
        self.enable_h2h = True if h2h != -1 and (in_c_h >= 0 and c_h > 0) else False
        if groups == (in_c_h + in_c_l):  # depthwise convolution
            assert c_l == in_c_l and c_h == in_c_h
            self.enable_l2h, self.enable_h2l = False, False
            is_dw = True
        use_bias_l2l, use_bias_h2l = (False, use_bias) if self.enable_h2l else (use_bias, False)
        use_bias_l2h, use_bias_h2h = (False, use_bias) if self.enable_h2h else (use_bias, False)
        s = (strides, strides) if type(strides) is int else strides
        do_stride2 = s[0] > 1 or s[1] > 1
        self.conv_l2l = None if not self.enable_l2l else nn.Conv2D(
            out_channels=c_l, kernel_size=kernel_size, stride=1,
            padding=padding, groups=groups if not is_dw else in_c_l,
            bias_attr=use_bias_l2l, in_channels=in_c_l)

        self.conv_l2h = None if not self.enable_l2h else nn.Conv2D(
            out_channels=c_h, kernel_size=kernel_size, stride=1,
            padding=padding, groups=groups,
            bias_attr=use_bias_l2h, in_channels=in_c_l, )

        self.conv_h2l = None if not self.enable_h2l else nn.Conv2D(
            out_channels=c_l, kernel_size=kernel_size, stride=1,
            padding=padding, groups=groups,
            bias_attr=use_bias_h2l, in_channels=in_c_h)

        self.conv_h2h = None if not self.enable_h2h else nn.Conv2D(
            out_channels=c_h, kernel_size=kernel_size, stride=1,
            padding=padding, groups=groups if not is_dw else in_c_h,
            bias_attr=use_bias_h2h, in_channels=in_c_h)

        self.l2l_down = nn.Identity() if not self.enable_l2l or not do_stride2 else \
            nn.AvgPool2D(strides, strides, ceil_mode=True)

        self.l2h_up = nn.Identity() if not self.enable_l2h or do_stride2 else \
            nn.Upsample(scale_factor=2, mode='nearest')
        self.h2h_down = nn.Identity() if not self.enable_h2h or not do_stride2 else \
            nn.AvgPool2D(strides, strides, ceil_mode=True)

        self.h2l_down = nn.Identity() if not self.enable_h2l else \
            nn.AvgPool2D((2 * s[0], 2 * s[1]), (2 * s[0], 2 * s[1]), ceil_mode=True)

    def _sum(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return x1 + x2
        else:
            return x1 if x2 is None else x2

    def forward(self, x_high, x_low=None):
        x_h2h = self.conv_h2h(self.h2h_down(x_high)) if self.enable_h2h else None
        x_h2l = self.conv_h2l(self.h2l_down(x_high)) if self.enable_h2l else None

        x_l2h = self.l2h_up(self.conv_l2h(x_low)) if self.enable_l2h else None
        x_l2l = self.conv_l2l(self.l2l_down(x_low)) if self.enable_l2l else None

        x_h = self._sum(x_l2h, x_h2h)
        x_l = self._sum(x_l2l, x_h2l)
        return (x_h, x_l)


class BatchNorm(nn.Layer):
    def __init__(self, in_channels=0):
        super().__init__()
        # be compatible to conventional convolution
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph, but got {}".format(
                in_channels)

        self.bn_h = nn.BatchNorm(in_c_h) if in_c_h >= 0 else lambda x: (x)
        self.bn_l = nn.BatchNorm(in_c_l) if in_c_l >= 0 else lambda x: (x)

    def forward(self, x_h, x_l=None):
        x_h = self.bn_h(x_h) if x_h is not None else None
        x_l = self.bn_l(x_l) if x_l is not None else None
        return (x_h, x_l)


class ReLU6(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x_h, x_l=None):
        x_h = F.relu6(x_h) if x_h is not None else None
        x_l = F.relu6(x_l) if x_l is not None else None
        return (x_h, x_l)


class BottleneckV1(nn.Layer):
    """ResNetV1 BottleneckV1
    """
    # pylint: disable=unused-argument
    def __init__(self, in_planes, mid_planes, out_planes, strides=1, ):
        super().__init__()

        self.use_shortcut = strides == 1 and in_planes == out_planes

        num_group = sum((c if c > 0 else 0 for c in mid_planes))

        # extract information
        self.conv1 = OctConv(out_channels=mid_planes, in_channels=in_planes,
                             kernel_size=1, use_bias=False)
        self.bn1 = BatchNorm(in_channels=mid_planes)
        self.relu1 = ReLU6()
        # capture spatial relations
        self.conv2 = OctConv(out_channels=mid_planes, in_channels=mid_planes,
                             kernel_size=3, padding=1, groups=num_group,
                             strides=strides, use_bias=False)
        self.bn2 = BatchNorm(in_channels=mid_planes)
        self.relu2 = ReLU6()
        # embeding back to information highway
        self.conv3 = OctConv(out_channels=out_planes, in_channels=mid_planes,
                             kernel_size=1, use_bias=False)
        self.bn3 = BatchNorm(in_channels=out_planes)

    def _sum(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return x1 + x2
        else:
            return x1 if x2 is None else x2

    def forward(self, x1, x2=None):
        x = (x1, x2)
        shortcut = x
        out = self.relu1(*self.bn1(*self.conv1(*x)))
        out = self.relu2(*self.bn2(*self.conv2(*out)))
        out = self.bn3(*self.conv3(*out))

        if self.use_shortcut:
            out = (self._sum(out[0], shortcut[0]), self._sum(out[1], shortcut[1]))

        return out


class MobileNetV2(nn.Layer):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.
    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, ratio=0., final_drop=0.):
        super().__init__()
        # reference:
        # - Howard, Andrew G., et al.
        #   "Mobilenets: Efficient convolutional neural networks for mobile vision applications."
        #   arXiv preprint arXiv:1704.04861 (2017).
        in_channels = [int(multiplier * x) for x in
                       [32] + [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
        mid_channels = [int(t * x) for t, x in zip([1] + [6] * 16, in_channels)]
        out_channels = [int(multiplier * x) for t, x in zip([1] + [6] * 16,
                                                            [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [
                                                                160] * 3 + [320])]
        strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3
        in_ratios = [0.] + [ratio] * 13 + [0.] * 3
        ratios = [ratio] * 13 + [0.] * 4
        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        self.conv1 = nn.Sequential(nn.Conv2D(3, int(32 * multiplier),
                                             kernel_size=3, padding=1, stride=2, bias_attr=False),
                                   nn.BatchNorm(int(32 * multiplier)),
                                   nn.ReLU6())

        stage_index, i = 1, 0
        for k, (in_c, mid_c, out_c, s, ir, r) in enumerate(
                zip(in_channels, mid_channels, out_channels, strides, in_ratios, ratios)):
            stage_index += 1 if s > 1 else 0
            i = 0 if s > 1 else (i + 1)
            name = 'L%d_B%d' % (stage_index, i)
            # -------------------------------------
            in_c = (in_c, -1)
            mid_c = self._get_channles(mid_c, r)
            out_c = (out_c, -1)
            # -------------------------------------
            setattr(self, name, BottleneckV1(in_c, mid_c, out_c, strides=s))
        # ------------------------------------------------------------------
        self.head = nn.Sequential(nn.Conv2D(out_channels=last_channels, in_channels=out_channels[-1],
                                            kernel_size=1, bias_attr=False),
                                  nn.BatchNorm(last_channels),
                                  nn.ReLU6(),
                                  nn.AdaptiveAvgPool2D(1),
                                  nn.Dropout(final_drop),
                                  nn.Conv2D(in_channels=last_channels, out_channels=classes,
                                            kernel_size=1)
                                  )

    def _get_channles(self, width, ratio):
        width = (width - int(ratio * width), int(ratio * width))
        width = tuple(c if c != 0 else -1 for c in width)
        return width

    def _concat(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return paddle.concat([x1, x2], axis=1)
        else:
            return x1 if x2 is None else x2

    def forward(self, x):
        x = self.conv1(x)
        x = (x, None)
        for iy in range(1, 10):
            # assume the max number of blocks is 50 per stage
            for ib in range(0, 50):
                name = 'L%d_B%d' % (iy, ib)
                if hasattr(self, name):
                    x = getattr(self, name)(*x)

        x = self.head(x[0])

        return x.flatten(1)


def OctMobileNetV2_x1_125(pretrained=False,
                          use_ssld=False,
                          **kwargs):
    model = MobileNetV2(1.125, ratio=0.5)
    return model


if __name__ == '__main__':
    model = OctMobileNetV2_x1_125()
    print(model)
    inp = paddle.rand([2, 3, 16, 16])
    # print(model(inp).shape)
    # print(paddle.summary(model, (2, 3, 16, 16)))
    # from paddle.vision import MobileNetV2
    # model = MobileNetV2(1.125)
    # print(paddle.summary(model, (2, 3, 16, 16)))