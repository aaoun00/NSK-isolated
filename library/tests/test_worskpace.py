
#TODO: set up gin ssh key for https://gin.g-node.org/

#TODO: load test data to the gin repository

#TODO: figure out how to read the data directly from the internet.

# eventually replace this with urllib
# access to the data online

from csv import DictReader
import os
import sys
from turtle import pos


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

 
from library.workspace import Workspace


def test_workspace():
    workspace = Workspace()

    assert isinstance(workspace, Workspace)