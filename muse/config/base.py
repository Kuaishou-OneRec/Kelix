# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base configuration class for all configs."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration class with common functionality.
    
    All configuration classes should inherit from this base class.
    Provides serialization, validation, and merging capabilities.
    """
    
    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields
        validate_assignment=True,  # Validate on assignment
        use_enum_values=True,  # Use enum values
        arbitrary_types_allowed=True,  # Allow arbitrary types
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        return self.model_dump()
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize config to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, path: str) -> None:
        """Save config to a JSON file.
        
        Args:
            path: File path to save the config
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    def merge(self, other: Dict[str, Any]) -> "BaseConfig":
        """Merge this config with another config dict.
        
        Values in 'other' will override values in this config.
        
        Args:
            other: Dictionary to merge with
            
        Returns:
            New config instance with merged values
        """
        current_dict = self.to_dict()
        current_dict.update(other)
        return self.__class__(**current_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary containing config values
            
        Returns:
            Config instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> "BaseConfig":
        """Create config from JSON string.
        
        Args:
            json_str: JSON string containing config values
            
        Returns:
            Config instance
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def load(cls, path: str) -> "BaseConfig":
        """Load config from a JSON file.
        
        Args:
            path: File path to load the config from
            
        Returns:
            Config instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

