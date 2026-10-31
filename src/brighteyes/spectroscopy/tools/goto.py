# -*- coding: utf-8 -*-
# Origin: VicidominiLab/BrightEyes-FFS, src/brighteyes_ffs/tools/goto.py at f6d640d.
"""
Change current working directory to the folder given as input
"""

import os
import platform

def goto(directory):
    if directory == 'data':
        path = 'SPAD-fcs\\Data\\'
    elif directory == 'fcs':
        path = 'Python\\fcs\\'
    elif directory == 'tools':
        path = 'Python\\tools\\'
    else:
        path = directory
    
    pf = platform.node()
    if pf == 'IITLW1768':
        # IIT laptop
        basePath = 'C:\\Users\\eslenders\\OneDrive\\OneDrive - Fondazione Istituto Italiano Tecnologia\\'
    elif pf == 'IITMMSDW002':
        # IIT desktop pc
        basePath = 'C:\\Users\\eslenders\\OneDrive - Fondazione Istituto Italiano Tecnologia\\'
    else:
        # lab PC
        basePath = 'C:\\Users\\SPAD-fcs\\OneDrive - Fondazione Istituto Italiano Tecnologia\\'
        
    os.chdir(basePath + path)
