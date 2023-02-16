# visdrone-det-toolkit-python

Python implementation of evaluation utilities of **[VisDrone2018-DET-toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)**. 

### Run Evaluation

Modify the dataset and result directories in evaluate.py and run:

```shell
python evaluate.py
```

### Installation and Usage

Installation:

```bash
pip install -e .
```

An example of using the function `eval_det` is given below:

```python
from viseval import eval_det
...
ap_all, ap_50, ap_75, ar_1, ar_10, ar_100, ar_500 = eval_det(
    annotations, results, heights, widths)
...
```

Reference: https://github.com/tjiiv-cprg/visdrone-det-toolkit-python.git

