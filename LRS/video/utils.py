import importlib

def check_availability(package_name: str) -> bool:
  try:
      importlib.import_module(package_name)
      print(f"The package '{package_name}' is installed.")
      return True
  except ImportError:
      print(f"The package '{package_name}' is not installed.")
      return False
