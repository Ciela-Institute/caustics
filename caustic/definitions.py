from collections import OrderedDict
import pprint



class NamespaceDict(OrderedDict):
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
    

class NestedNamespaceDict(NamespaceDict):
    def __getattr__(self, key):
        if key in self:
            value = self[key]
            if isinstance(value, dict):
                return NestedNamespaceDict(value)
            else:
                return value
        else:
            raise AttributeError(f"'NestedNamespaceDict' object has no attribute '{key}'")
    
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
        def _flatten_dict(dictionary, parent_key=""):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    _flatten_dict(value, key)
                else:
                    flattened_dict[key] = value
        _flatten_dict(self)
        return flattened_dict



        
