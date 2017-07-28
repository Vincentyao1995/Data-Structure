from setuptools import setup, find_packages
import os

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "dvm",
	version = "0.0.1",
	author = "David, Vincent, Matthew Dirks",
	author_email = "",
	description = ("Globalink 2017 project"),
	keywords = "",
	url = "",
	packages=find_packages(),
	# long_description=read('README'),
)
