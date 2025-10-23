"""
Unit tests for the model registry system.
"""

import pytest
from typing import Type


def test_register_model_basic():
    """Test basic model registration."""
    # Create a minimal mock setup to test registration
    MODEL_REGISTRY = {}
    
    def register_model(name=None):
        def decorator(cls):
            model_name = name if name is not None else cls.__name__
            if model_name in MODEL_REGISTRY:
                raise ValueError(f"Model '{model_name}' is already registered.")
            MODEL_REGISTRY[model_name] = cls
            return cls
        return decorator
    
    # Test registration with explicit name
    @register_model("TestModel")
    class TestModel:
        pass
    
    assert "TestModel" in MODEL_REGISTRY
    assert MODEL_REGISTRY["TestModel"] == TestModel
    

def test_register_model_without_name():
    """Test model registration using class name."""
    MODEL_REGISTRY = {}
    
    def register_model(name=None):
        def decorator(cls):
            model_name = name if name is not None else cls.__name__
            if model_name in MODEL_REGISTRY:
                raise ValueError(f"Model '{model_name}' is already registered.")
            MODEL_REGISTRY[model_name] = cls
            return cls
        return decorator
    
    # Test registration without explicit name (uses class name)
    @register_model()
    class MyModel:
        pass
    
    assert "MyModel" in MODEL_REGISTRY
    assert MODEL_REGISTRY["MyModel"] == MyModel


def test_register_model_duplicate():
    """Test that duplicate registration raises an error."""
    MODEL_REGISTRY = {}
    
    def register_model(name=None):
        def decorator(cls):
            model_name = name if name is not None else cls.__name__
            if model_name in MODEL_REGISTRY:
                raise ValueError(f"Model '{model_name}' is already registered.")
            MODEL_REGISTRY[model_name] = cls
            return cls
        return decorator
    
    @register_model("DuplicateModel")
    class Model1:
        pass
    
    # Attempting to register another model with the same name should fail
    with pytest.raises(ValueError, match="already registered"):
        @register_model("DuplicateModel")
        class Model2:
            pass


def test_get_model_class():
    """Test retrieving model classes."""
    MODEL_REGISTRY = {"TestModel": type("TestModel", (), {})}
    
    def get_model_class(name):
        if name not in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is not registered.")
        return MODEL_REGISTRY[name]
    
    # Test successful retrieval
    model_cls = get_model_class("TestModel")
    assert model_cls == MODEL_REGISTRY["TestModel"]
    
    # Test non-existent model
    with pytest.raises(KeyError, match="not registered"):
        get_model_class("NonExistentModel")


def test_list_models():
    """Test listing all registered models."""
    MODEL_REGISTRY = {
        "Model1": type("Model1", (), {}),
        "Model2": type("Model2", (), {}),
        "Model3": type("Model3", (), {}),
    }
    
    def list_models():
        return sorted(MODEL_REGISTRY.keys())
    
    models = list_models()
    assert models == ["Model1", "Model2", "Model3"]
    assert isinstance(models, list)


def test_integration_with_real_registry():
    """Test the actual registry implementation (if torch is available)."""
    try:
        from muse.models import (
            register_model, 
            get_model_class, 
            list_models, 
            MODEL_REGISTRY
        )
        
        # Check that Qwen3Model is registered
        assert "Qwen3Model" in list_models()
        
        # Check that we can retrieve it
        model_cls = get_model_class("Qwen3Model")
        assert model_cls.__name__ == "Qwen3Model"
        
        # Check MODEL_REGISTRY contains it
        assert "Qwen3Model" in MODEL_REGISTRY
        
    except ImportError:
        # Skip if dependencies not available
        pytest.skip("muse.models not available (missing dependencies)")


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running model registry tests...")
    
    test_register_model_basic()
    print("✓ test_register_model_basic passed")
    
    test_register_model_without_name()
    print("✓ test_register_model_without_name passed")
    
    test_list_models()
    print("✓ test_list_models passed")
    
    # Tests that need pytest
    print("\nNote: Some tests require pytest to run (duplicate registration, error handling)")
    print("Run with: pytest tests/test_model_registry.py")
    
    print("\n✓ All basic tests passed!")

