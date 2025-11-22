"""
Unit tests for muse.data.templates module.
"""
import pytest
import os
import tempfile
from pathlib import Path

from muse.data.templates import PromptLoader, TemplateLoader


class TestPromptLoader:
    """Test PromptLoader class"""

    def test_prompt_loader_init_default(self):
        """Test PromptLoader initialization with default library directory"""
        loader = PromptLoader()
        assert loader.library_dir is not None
        assert os.path.exists(loader.library_dir)

    def test_prompt_loader_init_custom(self, tmp_path):
        """Test PromptLoader initialization with custom library directory"""
        custom_dir = tmp_path / "custom_prompts"
        custom_dir.mkdir()
        loader = PromptLoader(library_dir=str(custom_dir))
        assert loader.library_dir == str(custom_dir)

    def test_prompt_loader_load_from_file(self, tmp_path):
        """Test loading prompt from file path"""
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_content = "This is a test prompt."
        prompt_file.write_text(prompt_content)

        loader = PromptLoader()
        result = loader.load(str(prompt_file))
        assert result == prompt_content

    def test_prompt_loader_load_from_library(self, tmp_path):
        """Test loading prompt from library directory"""
        library_dir = tmp_path / "prompts"
        library_dir.mkdir()
        prompt_file = library_dir / "default.txt"
        prompt_content = "Default prompt content"
        prompt_file.write_text(prompt_content)

        loader = PromptLoader(library_dir=str(library_dir))
        result = loader.load("default")
        assert result == prompt_content

    def test_prompt_loader_load_with_extension(self, tmp_path):
        """Test loading prompt with .txt extension"""
        library_dir = tmp_path / "prompts"
        library_dir.mkdir()
        prompt_file = library_dir / "test.txt"
        prompt_content = "Test content"
        prompt_file.write_text(prompt_content)

        loader = PromptLoader(library_dir=str(library_dir))
        result = loader.load("test.txt")
        assert result == prompt_content

    def test_prompt_loader_load_nonexistent(self, tmp_path):
        """Test loading non-existent prompt file"""
        library_dir = tmp_path / "prompts"
        library_dir.mkdir()
        loader = PromptLoader(library_dir=str(library_dir))

        result = loader.load("nonexistent")
        assert result == "nonexistent"  # Returns raw string if not found

    def test_prompt_loader_load_none(self):
        """Test loading with None input"""
        loader = PromptLoader()
        result = loader.load(None)
        assert result is None

    def test_prompt_loader_load_direct_string(self):
        """Test loading with direct string (not a file)"""
        loader = PromptLoader()
        result = loader.load("direct prompt string")
        assert result == "direct prompt string"


class TestTemplateLoader:
    """Test TemplateLoader class"""

    def test_template_loader_init_default(self):
        """Test TemplateLoader initialization with default library directory"""
        loader = TemplateLoader()
        assert loader.library_dir is not None
        # library_dir may not exist, but should be set to templates subdirectory
        assert "templates" in loader.library_dir

    def test_template_loader_init_custom(self, tmp_path):
        """Test TemplateLoader initialization with custom library directory"""
        custom_dir = tmp_path / "custom_templates"
        custom_dir.mkdir()
        loader = TemplateLoader(library_dir=str(custom_dir))
        assert loader.library_dir == str(custom_dir)

    def test_template_loader_load_from_file(self, tmp_path):
        """Test loading template from file path"""
        template_file = tmp_path / "test_template.jinja"
        template_content = "Hello {{ name }}!"
        template_file.write_text(template_content)

        loader = TemplateLoader()
        result = loader.load(str(template_file))
        assert result == template_content

    def test_template_loader_load_from_library(self, tmp_path):
        """Test loading template from library directory"""
        library_dir = tmp_path / "templates"
        library_dir.mkdir()
        template_file = library_dir / "chat.jinja"
        template_content = "Chat template: {{ message }}"
        template_file.write_text(template_content)

        loader = TemplateLoader(library_dir=str(library_dir))
        result = loader.load("chat")
        assert result == template_content

    def test_template_loader_load_with_extension(self, tmp_path):
        """Test loading template with .jinja extension"""
        library_dir = tmp_path / "templates"
        library_dir.mkdir()
        template_file = library_dir / "test.jinja"
        template_content = "Test template"
        template_file.write_text(template_content)

        loader = TemplateLoader(library_dir=str(library_dir))
        result = loader.load("test.jinja")
        assert result == template_content

    def test_template_loader_load_nonexistent(self, tmp_path):
        """Test loading non-existent template file"""
        library_dir = tmp_path / "templates"
        library_dir.mkdir()
        loader = TemplateLoader(library_dir=str(library_dir))

        result = loader.load("nonexistent")
        assert result == "nonexistent"  # Returns raw string if not found

    def test_template_loader_load_none(self):
        """Test loading with None input"""
        loader = TemplateLoader()
        result = loader.load(None)
        assert result is None

    def test_template_loader_load_direct_string(self):
        """Test loading with direct string (not a file)"""
        loader = TemplateLoader()
        result = loader.load("direct template string")
        assert result == "direct template string"

