from cgitb import html

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.imgmath', 
    'sphinx.ext.todo',
    'breathe',
    ]

html_theme = "furo"

# Breathe configuration
breathe_default_project = "cajal"