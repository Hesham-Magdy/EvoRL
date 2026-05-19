# Installation

1. install [Python](https://www.python.org/downloads/)

2. install [JAX](https://github.com/jax-ml/jax#installation)
   
   - this step is necessary for using JAX on GPU

3. download the source code to your device

4. change the current directory to the downloaded source code

5. run the following command:
   
   - ```
     python3 -m pip install --upgrade -r "./EvoRL/requirements.txt"
     ```

# Usage

### **List all the available algorithms in the framework**

```
python3 ./EvoRL/docs.py list algorithms
```

### **List all the available environments in the framework**

```
python3 ./EvoRL/docs.py list envs
```

### **List all the available activation functions in the framework**

```
python3 ./EvoRL/docs.py list activation_fn
```

### **Generate an input config file with a specific algorithm and environment**

- below is an example of generating an input config file with `ARS` algorithm and `CartPole` environment, and saving the generated configs to the `input.yaml` file

```
python3 ./EvoRL/docs.py generate ARS CartPole >> input.yaml
```

### **Run an experiment with the provided config file**

- open the `input.yaml` file and change any parameters you want before running the command

- the result of the experiment will be stored in `results` folder

```
python3 ./EvoRL/main.py input.yaml
```
