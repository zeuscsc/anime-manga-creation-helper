def import_or_install(package,pip_name=None):
    import importlib
    import subprocess
    if pip_name is None:
        pip_name=package
    try:
        importlib.import_module(package)
        # print(f"{package} is already installed")
    except ImportError:
        print(f"{package} is not installed, installing now...")
        subprocess.call(['pip', 'install', package])
        print(f"{package} has been installed")
