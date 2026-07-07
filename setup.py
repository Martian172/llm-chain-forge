"""
Setup configuration for LLM Chain Forge.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def parse_requirements(filename: str) -> list[str]:
    """Parse requirements from file, ignoring comments and empty lines."""
    requirements = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


setup(
    name="llm-chain-forge",
    version="0.1.0",
    author="LLM Chain Forge Contributors",
    author_email="hello@llm-chain-forge.dev",
    description="A lightweight yet powerful Python framework for building, testing, and optimizing LLM prompt chains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-chain-forge",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/llm-chain-forge/issues",
        "Documentation": "https://llm-chain-forge.readthedocs.io",
        "Discord": "https://discord.gg/llm-chain-forge",
        "Changelog": "https://github.com/yourusername/llm-chain-forge/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    package_data={
        "chainforge": [
            "playground/templates/*.html",
            "playground/static/**/*",
        ],
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.20.0",
        "httpx>=0.25.0",
        "pydantic>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "jinja2>=3.1.0",
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "tiktoken>=0.5.0",
        "diskcache>=5.6.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "aiohttp>=3.9.0",
        "pyyaml>=6.0.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "openai": ["openai>=1.0.0", "tiktoken>=0.5.0"],
        "anthropic": ["anthropic>=0.20.0"],
        "google": ["google-generativeai>=0.3.0"],
        "ollama": ["httpx>=0.25.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.20.0",
            "google-generativeai>=0.3.0",
            "tiktoken>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "forge=chainforge.cli.commands:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "llm", "prompt", "chain", "openai", "anthropic", "claude", "gpt",
        "ai", "nlp", "evaluation", "testing", "playground", "langchain-alternative"
    ],
)
