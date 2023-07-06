from collections import OrderedDict
import pprint


class NamespaceDict(OrderedDict):
    """
    Add support for attributes on top of an OrderedDict
    """
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(f"'NamespaceDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'NamespaceDict' object has no attribute '{key}'")

    def __repr__(self):
        return pprint.pformat(dict(self))

    def __str__(self):
        return pprint.pformat(dict(self))


class _NestedNamespaceDict(NamespaceDict):
    """
    Abstract method for NestedNamespaceDict and its Proxy
    """
    def flatten(self) -> NamespaceDict:
        """
        Flatten the nested dictionary into a NamespaceDict
        
        Returns:
            NamespaceDict: Flattened dictionary as a NamespaceDict
        """
        flattened_dict = NamespaceDict()
        def _flatten_dict(dictionary, parent_key=""):
            for key, value in dictionary.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, dict):
                    _flatten_dict(value, new_key)
                else:
                    flattened_dict[new_key] = value
        _flatten_dict(self)
        return flattened_dict

    def collapse(self) -> NamespaceDict:
        """
        Flatten the nested dictionary and collapse keys into the first level 
        of the NamespaceDict
        
        Returns:
            NamespaceDict: Flattened dictionary as a NamespaceDict
        """
        flattened_dict = NamespaceDict()
        def _flatten_dict(dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    _flatten_dict(value)
                else:
                    flattened_dict[key] = value
        _flatten_dict(self)
        return flattened_dict


class _NestedNamespaceProxy(_NestedNamespaceDict):
    """
    Proxy for NestedNamespaceDict in order to allow recursion in
    the class attributes
    """
    def __init__(self, parent, key_path):
        # Add new private keys to give us a ladder back to root node
        self._parent = parent
        self._key_path = key_path
        super().__init__(parent[key_path])

    def __setattr__(self, key, value):
        if key.startswith('_'):
            # We are in a child node, we need to recurse up
            super().__setattr__(key, value)
        else:
            # We are at the root node, call the __setitem__ to set record value
            self._parent.__setitem__(f"{self._key_path}.{key}", value)

    # Hide the private keys from common usage
    def keys(self):
        return [key for key in super().keys() if not key.startswith("_")]
            
    def items(self):
        for key, value in super().items():
            if not key.startswith('_'):
                yield (key, value)

    def values(self):
        return [v for k, v in super().items() if not k.startswith("_")]
    
    def __len__(self):
        # make sure hidden keys don't count in the length of the object
        return len(self.keys())


class NestedNamespaceDict(_NestedNamespaceDict):
    """
    Example usage
    ```python 
        nested_namespace = NestedNamespaceDict()
        nested_namespace.foo = 'Hello'
        nested_namespace.bar = {'baz': 'World'}
        nested_namespace.bar.qux = 42
        # works also in the follwoing way
         nested_namespace["bar.qux"] = 42

        print(nested_namespace)
        # Output:
        # {'foo': 'Hello', 'bar': {'baz': 'World', 'qux': 42 }}
        
        #==============================
        # Flattened key access
        #==============================
        print(nested_dict['foo'])        # Output: Hello
        print(nested_dict['bar.baz'])    # Output: World
        print(nested_dict['bar.qux'])    # Output: 42
        
        #==============================
        # Nested namespace access
        #==============================
        print(nested_dict.bar.qux)       # Output: 42
        
        #==============================
        # Flatten and collapse method
        #==============================
        print(nested_dict.flatten())
        # Output:
        # {'foo': 'Hello', 'bar.baz': 'World', 'bar.qux': 42}
        
        print(nested_dict.collapse()
        # Output:
        # {'foo': 'Hello', 'baz': 'World', 'qux': 42}
    
    """
    def __getattr__(self, key):
        if key in self:
            value = super().__getitem__(key)
            if isinstance(value, dict):
                return _NestedNamespaceProxy(self, key)
            else:
                return value
        else:
            raise AttributeError(f"'NestedNamespaceDict' object has no attribute '{key}'")

    def __getitem__(self, key):
        if "." in key:
            root, childs = key.split(".", 1)
            if root not in self:
                raise KeyError(f"'NestedNamespaceDict' object has no key '{key}'")
            return self[root].__getitem__(childs)
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, NestedNamespaceDict):
            value = NestedNamespaceDict(value)
        if "." in key:
            root, childs = key.split(".", 1)
            if root not in self:
                self[root] = NestedNamespaceDict()
            elif not isinstance(self[root], dict):
                raise ValueError("Can't assign a NestedNamespaceDict to a non-dict entry")
            self[root].__setitem__(childs, value)
        else:
            super().__setitem__(key, value)

