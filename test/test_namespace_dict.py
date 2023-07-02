from caustic.namespace_dict import NestedNamespaceDict
from pytest import raises


def test_namespace_dict__repr__():
    nested_namespace = NestedNamespaceDict()
    nested_namespace.foo = 1
    nested_namespace.bar = {"baz": 2}
    print(nested_namespace)


def test_namespace_dict_type():
    nested_namespace = NestedNamespaceDict()
    assert isinstance(nested_namespace, dict)


def test_nested_namespace_dict__setattr__():
    nested_namespace = NestedNamespaceDict()
    nested_namespace.foo = 1
    nested_namespace.bar = {"baz": 2}
    nested_namespace.bar.qux = 42
    print(nested_namespace)
    assert len(nested_namespace.keys()) == 2
    assert "foo" in nested_namespace
    assert "bar" in nested_namespace
    assert "baz" in nested_namespace.bar
    assert nested_namespace.bar.qux == 42


def test_nested_namespace_dict__setitem__():
    nested_namespace = NestedNamespaceDict()
    nested_namespace["foo"] = 1
    nested_namespace["bar"] = {"baz": 2}
    nested_namespace["bar.qux"] = {"quux": 42}
    nested_namespace["bar.qux.corge"] = 98
    print(nested_namespace)
    assert nested_namespace["foo"] == 1
    assert nested_namespace.foo == 1
    assert len(list(nested_namespace.keys())) == 2
    assert len(list(nested_namespace.bar.keys())) == 2
    assert len(list(nested_namespace.bar.values())) == 2
    assert len(list(nested_namespace.bar.items())) == 2
    assert nested_namespace["bar.qux.quux"] == 42
    assert nested_namespace.bar.qux.quux == 42
    assert nested_namespace["bar.qux.corge"] == 98
    assert nested_namespace.bar.qux.corge == 98


def test_nested_namespace_dict_flatten():
    nested_namespace = NestedNamespaceDict()
    nested_namespace["foo"] = 1
    nested_namespace["bar"] = {"baz": 2}
    nested_namespace["bar.qux"] = {"quux": 42}
    nested_namespace["bar.qux.corge"] = 98
    print(nested_namespace.flatten())
    assert len(list(nested_namespace.flatten().keys())) == 4


def test_nested_namespace_dict_collapse():
    nested_namespace = NestedNamespaceDict()
    nested_namespace["foo"] = 1
    nested_namespace["bar"] = {"baz": 2}
    nested_namespace["bar.qux"] = {"quux": 42}
    nested_namespace["bar.qux.corge"] = 98
    # name conflict, we keep the last one defined
    nested_namespace["bar.qux.foo"] = "Keep me"
    print(nested_namespace.collapse())
    assert nested_namespace.collapse().foo == "Keep me"
    assert len(list(nested_namespace.collapse().keys())) == 4


def test_nested_namespace_dict_collapse_shared_node():
    # Common DAG structure
    nested_namespace = NestedNamespaceDict()
    nested_namespace["foo"] = 1
    nested_namespace["bar"] = {"baz": 2}
    nested_namespace["bar.qux"] = {"quux": 42}
    nested_namespace["bar.qux.corge"] = 98



def test_nested_namespace_errors():
    nested_namespace = NestedNamespaceDict()
    with raises(AttributeError):
        print(nested_namespace.test)
    with raises(AttributeError):
        nested_namespace.foo = {"bar": 42}
        print(nested_namespace.foo.baz)
    with raises(AttributeError):
        nested_namespace.foo = 42
        nested_namespace.foo.bar = 98
    with raises(KeyError):
        nested_namespace.foo = {"bar": 42}
        print(nested_namespace["foo.baz"])
    with raises(KeyError):
        nested_namespace.foo = {"bar": 42}
        print(nested_namespace["qux.baz"])
    
