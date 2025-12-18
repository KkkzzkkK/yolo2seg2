from osgeo import gdal
from osgeo import osr
import os
import shutil

def ortho(file_name, dem_name, res, out_file_name):
    dataset = gdal.Open(file_name, gdal.GA_ReadOnly)
    is_north = 1 if os.path.basename(file_name).split('_')[3][0] == 'N' else 0
    zone = str(int(float(os.path.basename(file_name).split('_')[2][1:]) / 6) + 31)
    zone = int('326' + zone) if is_north else int('327' + zone)
    dstSRS = osr.SpatialReference()
    dstSRS.ImportFromEPSG(zone)
    tmp_ds = gdal.Warp(
        out_file_name,
        dataset,
        format='GTiff',
        xRes=res,
        yRes=res,
        dstSRS=dstSRS,
        rpc=True,
        resampleAlg=gdal.GRIORA_Bilinear,
        transformerOptions=["RPC_DEM=" + dem_name],
        srcNodata=0,                # 输入 nodata
        dstNodata=0,                # 输出 nodata
        dstAlpha=False
    )

    dataset = tds = None

def find_tif_files(folder_path):
    tif_files = []
    extensions = ("-NAD.tiff", "-MSS1.tiff", "-BWDMUX.tiff", "-MSS2.tiff","-MUX.tiff","-MSS.tiff")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extensions):
                tif_files.append(os.path.join(root, file))
    return tif_files

def process_files(GF_path, out_path, dem_name):
    for gf_file_name in os.listdir(GF_path):
        folder_path = os.path.join(GF_path, gf_file_name)
        if os.path.isdir(folder_path):
            tif_files = find_tif_files(folder_path)
            new_out_path = os.path.join(out_path, gf_file_name)
            if not os.path.exists(new_out_path):
                os.makedirs(new_out_path)
            for i in range(len(tif_files)):
                file_name = tif_files[i]
                print(f"[{file_name}] 开始处理...")
                xml_file = file_name.split('.tif')[0] + '.xml'
                shutil.copy(xml_file, new_out_path)
                out_file_name = os.path.join(new_out_path, os.path.basename(file_name))
                if 'GF1' in file_name:
                    if '-MSS' in file_name or '-MUX' in file_name:
                        res = 8
                    elif '-PAN' in file_name:
                        res = 4
                elif 'GF2' in file_name or 'GF7' in file_name:
                    if '-MSS' in file_name or '-MUX' in file_name:
                        res = 4
                    elif '-PAN' in file_name:
                        res = 1
                ortho(file_name, dem_name, res, out_file_name)
                print("图像已保存为:", out_file_name)

if __name__ == "__main__":
    # 主文件夹路径
    input_dir = r"/home/MXJ/data/sample_data"
    output_dir = r"/home/MXJ/data/sample_data_proj"
    dem_path = r"/home/MXJ/ultralystics-main/ultralytics-main/tif_tools/GMTED2010.jp2"
    if not os.path.exists(input_dir):
        print("输入目录不存在")

    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(output_nir_dir, exist_ok=True)

    extensions = ("-NAD.tiff", "-MSS1.tiff", "-BWDMUX.tiff", "-MSS2.tiff","-MUX.tiff","-MSS.tiff")
    # extensions = ()
    # 遍历主文件夹及其所有子文件夹
    # for dirpath, dirnames, filenames in os.walk(input_dir):
    #     for filename in filenames:
            # 检查文件名是否符合模式
    process_files(input_dir, output_dir, dem_path)