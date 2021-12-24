import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def ConvBlock(in_channels, channels, kernel_size, strides):
    return nn.Sequential(
        nn.Conv2D(in_channels, channels, kernel_size, stride=strides, padding=1, bias_attr=False),
        nn.BatchNorm(channels),
        nn.ReLU())


def Conv1x1(in_channels, channels, is_linear=False):
    return nn.Sequential(
        nn.Conv2D(in_channels, channels, 1),
        nn.BatchNorm(channels),
        nn.Identity() if is_linear else nn.ReLU())


def DWise(in_channels, channels, strides, kernel_size=3):
    return nn.Sequential(
        nn.Conv2D(in_channels, channels, kernel_size, stride=strides, padding=kernel_size // 2, groups=channels,
                  bias_attr=False),
        nn.BatchNorm(channels),
        nn.ReLU())


class SepCONV(nn.Layer):
    def __init__(self, inp, output, kernel_size, depth_multiplier=1, with_bn=True):
        super(SepCONV, self).__init__()
        self.net = []
        cn = int(inp * depth_multiplier)

        if output is None:
            self.net.append(
                nn.Conv2D(in_channels=inp, out_channels=cn, groups=inp, kernel_size=kernel_size, stride=(1, 1),
                          padding=kernel_size // 2
                          , bias_attr=not with_bn)
            )
        else:
            self.net.extend([
                nn.Conv2D(in_channels=inp, out_channels=cn, groups=inp, kernel_size=kernel_size, stride=(1, 1),
                          padding=kernel_size // 2
                          , bias_attr=False),
                nn.BatchNorm(cn),
                nn.ReLU(),
                nn.Conv2D(in_channels=cn, out_channels=output, kernel_size=(1, 1), stride=(1, 1)
                          , bias_attr=not with_bn)]
            )

            self.net = nn.Sequential(*self.net)
            self.with_bn = with_bn
            self.act = nn.ReLU()
            if with_bn:
                self.bn = nn.BatchNorm(output) if output is not None else nn.BatchNorm(cn)

    def forward(self, x):
        x = self.net(x)
        if self.with_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ExpandedConv(nn.Layer):
    def __init__(self, inp, oup, t, strides, kernel=3, same_shape=True):
        super(ExpandedConv, self).__init__()

        self.same_shape = same_shape
        self.strides = strides
        self.bottleneck = nn.Sequential(Conv1x1(inp, inp*t),
                                        DWise(inp*t, inp*t, self.strides, kernel),
                                        Conv1x1(inp*t, oup, is_linear=True))

    def forward(self, x):
        out = self.bottleneck(x)
        if self.strides == 1 and self.same_shape:
            out = out + x
        return out


def ExpandedConvSequence(t, k, inp, oup, repeats, first_strides):
    seq = [ExpandedConv(inp, oup, t, first_strides, k, same_shape=False)]
    curr_inp = oup
    for i in range(1, repeats):
        seq.append(ExpandedConv(curr_inp, oup, t, 1))
        curr_inp = oup
    return nn.Sequential(*seq)


class MNasNet(nn.Layer):
    def __init__(self, class_num=1000, **kwargs):
        super(MNasNet, self).__init__(**kwargs)

        self.first_oup = 32
        self.interverted_residual_setting = [
            # t, c,  n, s, k
            [3, 24, 3, 2, 3, "stage2_"],  # -> 56x56
            [3, 40, 3, 2, 5, "stage3_"],  # -> 28x28
            [6, 80, 3, 2, 5, "stage4_1_"],  # -> 14x14
            [6, 96, 2, 1, 3, "stage4_2_"],  # -> 14x14
            [6, 192, 4, 2, 5, "stage5_1_"],  # -> 7x7
            [6, 320, 1, 1, 3, "stage5_2_"],  # -> 7x7          
        ]
        self.last_channels = 1280

        self.features = []
        self.features.append(ConvBlock(3, self.first_oup, 3, 2,))
        self.features.append(SepCONV(self.first_oup, 16, 3, ))
        inp = 16
        for i, (t, c, n, s, k, prefix) in enumerate(self.interverted_residual_setting):
            oup = c
            self.features.append(ExpandedConvSequence(t, k, inp, oup, n, s, ))
            inp = oup

        self.features.append(Conv1x1(oup, self.last_channels))
        self.features.append(nn.AdaptiveAvgPool2D(1))
        self.features.append(nn.Flatten())
        self.features = nn.Sequential(*self.features)
        self.output = nn.Linear(self.last_channels, class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x

def mnasnet1_0_mxnet(**kwargs):
    return MNasNet(**kwargs)

if __name__ == '__main__':
    import paddle
    net = MNasNet(1000)
    # x = paddle.randn([1,3, 2, 2])
    # net(x)
    paddle.summary(net, (1, 3, 224, 224))

    # save as symbol
    # data =mx.sym.var('data')
    # sym = net(data)

    ## plot network graph
    # mx.viz.print_summary(sym, shape={'data':(8,3,224,224)})
    # mx.viz.plot_network(sym,shape={'data':(8,3,224,224)}, node_attrs={'shape':'oval','fixedsize':'fasl==false'}).view()
