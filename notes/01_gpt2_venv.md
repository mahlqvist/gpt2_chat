# Basic Setup

Start by creating a virtual environment to install the necessary python packages. To make the virtual environment available as a kernel in Jupyter Notebook, install **ipykernel**.

The **IPython kernel** is the python execution backend for Jupyter Notebook and **IPython** is just an interactive command-line terminal for python. 

```bash
python -m venv .myenv

pip install ipykernel

# Add your virtual env to Jupyter, replace .myenv with the name of your venv
python -m ipykernel install --user --name=.myenv
```

This should output the path to the folder where you will find the `kernel.json` file.

```ini
Installed kernelspec myenv in /home/user/.local/share/jupyter/kernels/.myenv
```

On Windows the path will look like this: 

```ini
C:\Users\username\AppData\Roaming\jupyter\kernels\.myenv
```

**Optional**: install `ipywidgets`, which is a python library that provides interactive html-widgets for Jupyter notebooks and JupyterLab. For example, the `tqdm` library is used for displaying progress bars and by installing `ipywidgets`, you enable `tqdm` to use a more advanced **IProgress widget**, improving the user experience in Jupyter notebooks and you won't get warnings about it.

```bash
# to install ipywidgets
pip install ipywidgets

# to list the kernels
jupyter kernelspec list

# to uninstall the kernel
jupyter kernelspec uninstall .myenv
```

Start Jupyter Notebook and select the virtual environment kernel. Once Jupyter Notebook opens in your web browser, create or open a notebook. In the notebook interface, go to the **Kernel** menu and select **Change kernel**. From the list of available kernels, choose the one corresponding to your virtual environment (.myenv).

`!pip install` is used within a **Jupyter Notebook** or the **IPython** shell, where the exclamation mark `!` is a special syntax that allows running shell commands directly from the notebook or shell.

> Remember to activate the virtual environment and start Jupyter Notebook with the activated environment each time you want to work with it