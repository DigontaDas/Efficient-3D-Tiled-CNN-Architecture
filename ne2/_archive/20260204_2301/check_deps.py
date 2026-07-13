import sys
try:
    import monai
    print('MONAI available:', monai.__version__)
except ImportError:
    print('MONAI not available')

try:
    import nnunetv2
    print('nnU-Net available')
except ImportError:
    print('nnU-Net not available')

try:
    import SimpleITK as sitk
    print('SimpleITK available:', sitk.__version__)
except ImportError:
    print('SimpleITK not available')

print('Python path:', sys.executable)
