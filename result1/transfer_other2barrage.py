import os
from tqdm import tqdm
import logging

def setup_logging(log_file='modify_dam_types.log'):
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def modify_dam_type_in_file(input_file, output_file, dam_type_index=9, 
                           old_type='other_dam', new_type='Barrage_dam'):
    """
    修改文本文件中的水坝类型。
    
    参数:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        dam_type_index (int): 水坝类型在行中的索引位置（从0开始）
        old_type (str): 要替换的原类型
        new_type (str): 新的类型
    """
    try:
        count_total = 0
        count_modified = 0
        
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            for line in tqdm(fin, desc=f"修改水坝类型 {old_type} -> {new_type}", unit="行"):
                count_total += 1
                parts = line.strip().split()
                
                # 确保行有足够的字段
                if len(parts) > dam_type_index:
                    # 检查并替换水坝类型
                    if parts[dam_type_index] == old_type:
                        parts[dam_type_index] = new_type
                        count_modified += 1
                
                # 写入修改后的行
                fout.write(' '.join(parts) + '\n')
        
        logging.info(f"处理完成: 共处理 {count_total} 行，修改了 {count_modified} 个 '{old_type}' 为 '{new_type}'")
        return count_modified
    except Exception as e:
        logging.error(f"修改文件时出错: {e}")
        return 0

def main():
    setup_logging()
    
    # 定义基础输入和输出目录
    base_input_dir = r'E:\wyj\dam\v4\global_v5'  # 替换为您的合并文本文件所在目录
    base_output_dir = r'E:\wyj\dam\v4\global_v5'  # 修改后的文件输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 定义要处理的年份列表
    years = [2010, 2015, 2020]
    
    # 定义水坝类型字段在行中的索引位置（从0开始）
    dam_type_index = 8  # 假设水坝类型在第9个位置（索引为8）
    
    # 定义要替换的类型
    old_type = 'other_dam'
    new_type = 'Barrage_dam'
    
    # 处理每个年份的文件
    for year in years:
        input_file = os.path.join(base_input_dir, f"all_{year}_geo_0.4_filtered_all_merged.txt")
        if not os.path.exists(input_file):
            logging.warning(f"输入文件不存在: {input_file}")
            continue
            
        output_file = os.path.join(base_output_dir, f"all_{year}_geo_0.4_filtered_all_merged_o2b.txt")
        
        logging.info(f"开始处理年份 {year} 的文件: {input_file}")
        modified_count = modify_dam_type_in_file(input_file, output_file, dam_type_index, old_type, new_type)
        
        if modified_count > 0:
            logging.info(f"已成功修改年份 {year} 的文件，保存为: {output_file}")
        else:
            logging.warning(f"年份 {year} 的文件中未找到 '{old_type}' 类型的水坝")

if __name__ == "__main__":
    main()