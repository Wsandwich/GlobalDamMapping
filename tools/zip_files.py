import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def compress_folder(folder_path, output_zip):
    """
    压缩指定文件夹中的所有文件到一个zip文件中，并显示进度条。
    
    :param folder_path: 要压缩的文件夹路径
    :param output_zip: 输出的zip文件路径
    """
    # 获取所有文件列表
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 生成相对于文件夹的路径，以保留文件结构
            file_paths.append(os.path.relpath(os.path.join(root, file), folder_path))
    
    # 创建zip文件
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 使用tqdm显示进度条
        for file in tqdm(file_paths, desc=f'压缩 {os.path.basename(folder_path)}', unit='文件'):
            full_path = os.path.join(folder_path, file)
            zipf.write(full_path, arcname=file)

def main():
    # 要压缩的两个文件夹路径
    folder1 = r"F:\study\project3\roLabelimg\global_clear\YANGTZE\labels_v2\xml\txt_label_v1"
    # folder2 = 'detection/mmrotate-main/out_global/INDUS_2020_v4_resnet/images_v2'
    
    # 输出的zip文件路径
    output_zip1 = r"F:\study\project3\roLabelimg\global_clear\YANGTZE\labels_v2\xml\txt_label_v1\YANGTZE下半_dota.zip"
    # output_zip2 = 'detection/mmrotate-main/out_global/INDUS_2020_v4_resnet/images_v2.zip'
    
    # 使用ThreadPoolExecutor进行多线程压缩
    with ThreadPoolExecutor(max_workers=10) as executor:
        future1 = executor.submit(compress_folder, folder1, output_zip1)
        # future2 = executor.submit(compress_folder, folder2, output_zip2)
        
        # 等待所有线程完成
        future1.result()
        # future2.result()
    
    print('所有文件夹已成功压缩。')

if __name__ == '__main__':
    main()
