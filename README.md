# dd2434-project
Project in DD2434 Machine Learning Advance Course, Winter 2016

## What did we do?
We reproduced the experiments presented in the paper [Kernel PCA and De-noising in Feature Spaces](docs/paper.pdf) by Sebastian Mika, Bernhard Schölkopf, Alex Smola Klaus-Robert Müller, Matthias Scholz and Gunnar Rätsch. In this regard, you can read our [report](docs/report.pdf) and our [presentation](docs/presentation.pdf)

## The experiments


In the paper, there are three major experiments:

* Toy example: 11 Gaussians
* Toy example: De-noising
* Digit denoising (USPS Dataset)

However, prior to

### Toy example: 11 Gaussians
The code related to this example can be found in [example1.py](example1.py).

Run the script as
```
python3 example1.py
```

By default, you should obtain the results (kPCA MSE, PCA PCA and their ratio) for 45 different settings of sigma.


### Toy example: De-noising
The code related to this example can be found in [example2.py](example2.py)

Run the script as
```
python3 example2.py
```

Once the execution has ended, a picture as follows should be displayed.

![alt text][img/figure]

You might get some warnings, just ignore them.
### Digit denoising (USPS Dataset)
The code related to this example can be found in [example3.py](example3.py)


Feel free to run the python files containing

### Virtualenv setup
Install virtualenv
``` bash
sudo apt install python3-venv
```
Create a virtualenv somewhere
``` bash
python3 -m venv <env name>
```
Activate the environment and move to the ```<repo folder>```
``` bash
. <env name>/bin/activate
```
The first time, install the packages from ```requirements.txt```
``` bash
pip install -r requirements.txt
```
(If you install something with ```pip install``` remember to dump the packages installed and push the new requirements)
``` bash
pip freeze > requirements.txt
```
Deactivate the environment
``` bash
deactivate
```

## Stuff to do
- Receive confirmation of paper acceptance
- Read [paper](1491-kernel-pca-and-de-noising-in-feature-spaces.pdf)
- Summarize the paper for our colleagues
- Implementation
- Create nice examples (data, plots...)
  - [Example 1](http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html)
  - [Example 2](http://sebastianraschka.com/Articles/2014_kernel_pca.html)
- Write report
- Prepare presentation slides and rehearse
- Sign up for presentation?

## Roadmap
| Deadline   | What                                     |
|------------|------------------------------------------|
| 22/12/2016 | Starting project                         |
| 24/12/2016 | Reading finished                         |
| 04/01/2017 | Summary ready                            |
| 04/01/2017 | Implementation ended                     |
| 07/01/2017 | Examples implementation                  |
| 10/01/2017 | 1st Draft of the report                  |
| 15/01/2017 | Presentation finished, Presentation test |
| 16/01/2017 | Day of the presentation (sign up?)       |

## Interesting links
- [Kernel Principal Component Analysis and its Applications in Face Recognition and Active Shape Models](https://arxiv.org/pdf/1207.3538v3.pdf)
- [Support Vector Learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.35.380&rep=rep1&type=pdf)
- [sklearn.decomposition.KernelPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)
