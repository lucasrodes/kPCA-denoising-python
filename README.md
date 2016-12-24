# dd2434-project
Project in DD2434 Machine Learning Advance Course, Winter 2016

## How we will work?

- We will work on the report in [Overleaf](www.overleaf.com)
- Coding related files will be stored here
- Presentation slides [Google]

## Virtualenv setup
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

## Stuff to do
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
