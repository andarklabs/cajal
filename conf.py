# for running with sphinx... unsure how well this will work...

from cgitb import html
import sys

sys.path.append("cajal/docs/ext/breathe/")

breathe_domain_by_extension = {
    "h" : "cpp",
}

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

breathe_projects = { "cajal": "cajal/docs/xml/" }

# Breathe configuration
breathe_default_project = "cajal"

master_doc = 'index'

