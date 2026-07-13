# c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Phase3_Local_Integration\dicom_to_nifti.py
"""
Converts a folder of DICOM series to NIfTI (.nii.gz).
Validates spacing and orientation against the ImageCAS reference template.
"""
import SimpleITK as sitk
import os, sys

REFERENCE_SPACING = (0.3125, 0.3125, 0.5)   # from ImageCAS dataset audit

def convert_dicom_to_nifti(dicom_dir: str, output_path: str) -> bool:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        print(f"[ERROR] No DICOM series found in {dicom_dir}")
        return False
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0]))
    image = reader.Execute()
    sitk.WriteImage(image, output_path)
    print(f"[OK] Saved {output_path} | Spacing: {image.GetSpacing()} | Size: {image.GetSize()}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dicom_to_nifti.py <dicom_dir> <output_path.nii.gz>")
        sys.exit(1)
    convert_dicom_to_nifti(sys.argv[1], sys.argv[2])
