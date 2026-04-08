import os
import importlib

"""
import sagemaker

helper = ImportHelper(sagemaker)
helper.explore()

helper.guess_import("ModelBuilder")

ModelBuilder = helper.try_import("ModelBuilder")
"""

class ImportHelper:
    def __init__(self, package):
        self.package = package
        self.package_name = package.__name__
        self.package_path = package.__path__[0]

    
    def _locate_class_in_package(self, class_name):
        matches = []
        for root, _, files in os.walk(self.package_path):
            for fname in files:
                if fname.endswith(".py"):
                    path = os.path.join(root, fname)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            for lineno, line in enumerate(f, start=1):
                                if f"class {class_name}" in line:
                                    matches.append((path, lineno))
                    except Exception:
                        pass
        return matches

    
    def _guess_module_from_path(self, file_path):
        idx = file_path.find(self.package_name)
        if idx == -1:
            return None
        module_path = file_path[idx:].replace(os.sep, ".")
        module_path = module_path.replace(".py", "")
        module_path = module_path.replace(".__init__", "")
        return module_path


    def guess_import(self, class_name):
        matches = self._locate_class_in_package(class_name)
        for path, lineno in matches:
            module_path = self._guess_module_from_path(path)
            if module_path:
                print(f"Found in: {path}:{lineno}")
                print(f"Likely import: from {module_path} import {class_name}")

    
    def try_import(self, class_name):
        matches = self._locate_class_in_package(class_name)
        for path, _ in matches:
            module_path = self._guess_module_from_path(path)
            if not module_path:
                continue
            try:
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except Exception:
                pass
        raise ImportError(f"Could not import {class_name}")


    def explore(self):
        module = self.package
        
        def list_module_tree(module_path, level=0, prefix=""):
            if os.path.isdir(module_path):
                if os.path.basename(module_path) == '__pycache__':
                    return
                
                print(f"{prefix}+-- {os.path.basename(module_path)}/")
                
                for item in os.listdir(module_path):
                    item_path = os.path.join(module_path, item)
                    
                    new_prefix = "   " * (level + 1)
                    list_module_tree(item_path, level + 1, new_prefix)
        
        module_path = os.path.dirname(module.__file__)
        
        list_module_tree(module_path, level=0, prefix="")
