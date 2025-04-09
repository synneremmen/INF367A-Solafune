import os
import rasterio
from zipfile import ZipFile
from glob import glob
import shutil

# 1. Set up paths relative to this script
base_dir = os.path.dirname(__file__)  # path to /utils
input_dir = os.path.abspath(os.path.join(base_dir, "../data/images"))
output_dir = os.path.abspath(os.path.join(base_dir, "../data/superres"))
temp_dir = os.path.join(base_dir, "temp_safes")

# 2. Make sure output folders exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# 3. Map bands to Sentinel-2 resolutions
band_resolution = {
    1: "R60m",
    2: "R10m",
    3: "R10m",
    4: "R10m",
    5: "R20m",
    6: "R20m",
    7: "R20m",
    8: "R10m",
    "8A": "R20m",
    9: "R60m",
    11: "R20m",
    12: "R20m"
}

def create_valid_metadata(safe_path, granule_id="L2A_T31UFT_A012345_20250101T103021"):
    """Generate a fully compliant MTD_MSIL2A.xml file."""
    mtd_path = os.path.join(safe_path, "MTD_MSIL2A.xml")
    
    xml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd">
  <n1:General_Info>
    <Product_Info>
      <PRODUCT_URI>{os.path.basename(safe_path)}</PRODUCT_URI>
      <PROCESSING_LEVEL>LEVEL2A</PROCESSING_LEVEL>
      <PRODUCT_TYPE>S2MSI2A</PRODUCT_TYPE>
      <PROCESSING_BASELINE>04.00</PROCESSING_BASELINE>
    </Product_Info>
  </n1:General_Info>
  <n1:L2A_Product_Organisation>
    <n1:L2A_Product_Info>
      <n1:Product_Components>
        <n1:Granules>
          <n1:Granule granuleIdentifier="{granule_id}">
            <n1:IMAGE_FILE>
              <n1:Band_ID>B01</n1:Band_ID>
              <n1:File_Location>GRANULE/{granule_id}/IMG_DATA/R60m/B01_60m.tif</n1:File_Location>
            </n1:IMAGE_FILE>
            <n1:IMAGE_FILE>
              <n1:Band_ID>B02</n1:Band_ID>
              <n1:File_Location>GRANULE/{granule_id}/IMG_DATA/R10m/B02_10m.tif</n1:File_Location>
            </n1:IMAGE_FILE>
            <!-- Add ALL bands your tool requires here -->
          </n1:Granule>
        </n1:Granules>
      </n1:Product_Components>
    </n1:L2A_Product_Info>
  </n1:L2A_Product_Organisation>
  <n1:Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_NAME>WGS84 / UTM zone 31N</HORIZONTAL_CS_NAME>
    </Tile_Geocoding>
  </n1:Geometric_Info>
</n1:Level-2A_User_Product>
"""
    with open(mtd_path, 'w') as f:
        f.write(xml_template.strip())

def create_fake_safe_structure(safe_path, granule_id):
    """Create folders with valid metadata."""
    # Root structure
    os.makedirs(os.path.join(safe_path, "GRANULE", granule_id, "IMG_DATA"), exist_ok=True)
    create_valid_metadata(safe_path, granule_id)

    # Subfolders for resolutions (R10m, R20m, R60m)
    for res in ['R10m', 'R20m', 'R60m']:
        os.makedirs(os.path.join(safe_path, "GRANULE", granule_id, "IMG_DATA", res), exist_ok=True)

# 4. Get all tif files
tif_files = glob(os.path.join(input_dir, "*.tif"))
if not tif_files:
    print("❌ No .tif files found.")
    exit()

# 5. Process each file
for tif_path in tif_files:
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    safe_name = f"{base_name}.SAFE"
    safe_path = os.path.join(temp_dir, safe_name)
    granule_id = "L2A_T31UFT_A012345_20250101T103021"  # Match this in XML
    create_fake_safe_structure(safe_path, granule_id)

    with rasterio.open(tif_path) as src:
        for i in range(1, src.count + 1):
            band = src.read(i)
            profile = src.profile
            profile.update(count=1)

            res = band_resolution.get(i, "R10m")           
            # Update band output names:
            out_path = os.path.join(safe_path, "GRANULE", granule_id, "IMG_DATA", res, f"B{i:02}_{res.replace('R', '')}m.tif"  # e.g., "B02_10m.tif"
                                    )

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(band, 1)

    # 6. Zip the .SAFE folder
    zip_name = f"{base_name}_SAFE.zip"
    zip_path = os.path.join(output_dir, zip_name)
    with ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(safe_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, temp_dir)
                zipf.write(full_path, rel_path)

    print(f"✅ Created zip: {zip_path}")

    # Optional: clean up temp
    shutil.rmtree(safe_path)

print("🎉 All files converted and saved in /data/superres/")
