import gdcm
import sys
import dicom
import pylab

if __name__ == "__main__":
  file1 = 'I00001264379.dcm' # input filename
  file2 = 'bone.dcm' # output filename (decompress)

  reader = gdcm.ImageReader()
  reader.SetFileName( file1 )

  if not reader.Read():
    sys.exit(1)

  change = gdcm.ImageChangeTransferSyntax()
  change.SetTransferSyntax( gdcm.TransferSyntax(gdcm.TransferSyntax.ImplicitVRLittleEndian) )
  change.SetInput( reader.GetImage() )
  if not change.Change():
    sys.exit(1)

  writer = gdcm.ImageWriter()
  writer.SetFileName( file2 )
  writer.SetFile( reader.GetFile() )
  writer.SetImage( change.GetOutput() )

  if not writer.Write():
    sys.exit(1)

  ds=dicom.read_file('bone.dcm')
  pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
  pylab.savefig('bone.png')
  pylab.show()
