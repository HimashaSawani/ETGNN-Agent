import zipfile
from pathlib import Path

z = zipfile.ZipFile("ETGNN_Colab_Bundle.zip")
print("Files in ETGNN_Colab_Bundle.zip:")
for name in z.namelist()[:20]:
    print("  ", name)

