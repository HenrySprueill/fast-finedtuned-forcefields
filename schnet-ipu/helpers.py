# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import functools
import torch.testing

assert_equal = functools.partial(torch.testing.assert_close, rtol=0., atol=0.)