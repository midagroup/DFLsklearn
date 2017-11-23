#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:59:25 2017

@authors: V. Latorre, F. Benvenuto

"""
import setuptools
from numpy.distutils.core import Extension
from numpy.distutils.core import setup
import numpy
import distutils


ext= Extension(name = 'dfl',
                 sources = ['./Fortran_codes/main_box.f90','./Fortran_codes/sd.f90','./Fortran_codes/dfl.pyf'])



if __name__ == "__main__":
    
    setup(name = 'DFL',
              description       = "Derivative free linesearch code",
              version="0.1",
              ext_modules = [ext]
              )
    setup(name="DFLsklearn",
     		version="0.1",
	  	author            = "Vittorio Latorre, Federico Benvenuto",
              	author_email      = "latorre@dima.unige.it, benvenuto@dima.unige.it",
             	py_modules=['DFLsklearn'],
             	zip_safe = False                    
             	)

