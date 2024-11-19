# __version__ = '0.0.7'
# __version__ = '0.0.8' Preliminary version
# __version__ = '0.1.0' ERC version on
# __version__ = '0.3.0' Pytorch version
__version__ = '0.9'    # Pytorch version with recurrent network and export to onnx (only not reccurent)
# __version__ = '1.0'  # Export onnx with recurrent

import sys
major, minor = sys.version_info.major, sys.version_info.minor

if major < 3:
    sys.exit("Sorry, Python 2 is not supported. You need Python >= 3.6 for "+__package__+".")
elif minor < 10:
    sys.exit("Sorry, You need Python >= 3.10 for "+__package__+".")
else:
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>---- '+ __package__+' ----<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')