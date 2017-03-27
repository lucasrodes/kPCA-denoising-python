# Set up a virtual environment

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