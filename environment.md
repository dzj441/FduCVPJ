# how to run our code in paddle  
there are some unexpected bugs in the paddle AI studio. To run our code,we suggest u to redeploy your environment as follows:  
1. enter the terminal and check paddle version 
```bash
pip show paddlepaddle  
``` 
if the version is 2.1.2, chances are that it's an incompatible version. uninstall it.  
```bash 
pip uninstall paddlepaddle
```

2. update pip 
```bash
pip install pip --upgrade
```
3. the paddle version maybe incompatible with the paddleseg version. we adopt version 2.3.0
```bash
python -m pip install paddlepaddle-gpu==2.3.0 -i https://mirror.baidu.com/pypi/simple
```
4. now u can download paddleseg. our version is 2.8.0
```bash
pip install paddleseg
```
5. restart the environment