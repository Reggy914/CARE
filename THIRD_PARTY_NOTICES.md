# Third-Party Notices

CARE is distributed under the MIT License. Its direct Python dependencies and
the external Timer checkpoint remain governed by their own licenses.

No third-party dependency source code, pretrained Timer weight, or Timer custom
model file is bundled in this repository. Dependencies are installed by the
user, and the Timer resource is downloaded separately.

| Component | Declared range or resource | Upstream license | Upstream license or resource |
|---|---|---|---|
| PyTorch | torch >=2.2,<3 | BSD-3-Clause; distributed builds can include separately licensed components | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| NumPy | numpy 2.2.6 | BSD-3-Clause; binary distributions can include separately licensed components | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| pandas | pandas 2.3.3 | BSD-3-Clause | https://github.com/pandas-dev/pandas/blob/main/LICENSE |
| scikit-learn | scikit-learn 1.7.2 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| PyYAML | PyYAML 6.0.3 | MIT | https://github.com/yaml/pyyaml/blob/main/LICENSE |
| Transformers | transformers 4.40.1 | Apache-2.0 | https://github.com/huggingface/transformers/blob/main/LICENSE |
| huggingface-hub | huggingface-hub 0.36.2 | Apache-2.0 | https://github.com/huggingface/huggingface_hub/blob/main/LICENSE |
| tabulate | tabulate >=0.9,<1 | MIT | https://github.com/astanin/python-tabulate/blob/master/LICENSE |
| Timer checkpoint and custom model files | thuml/timer-base-84m | Apache-2.0, as declared by the model repository | https://huggingface.co/thuml/timer-base-84m |

The table is an engineering inventory, not legal advice. The license files
shipped with the exact packages and resources a user downloads are
authoritative. Anyone redistributing dependencies, binary wheels, Timer
weights, or Timer custom model code must preserve the notices and satisfy the
terms supplied with those artifacts. Apache-2.0 redistribution may also
require preserving an upstream NOTICE file when one is provided.
