# 使用geopandas读取GeoPackage文件
import geopandas as gpd
import matplotlib.pyplot as plt

def read_geopackage(file_path):
    """
    读取GeoPackage文件并返回GeoDataFrame
    """
    try:
        # 使用geopandas读取GeoPackage文件
        gdf = gpd.read_file(file_path)
        print(f"成功读取文件: {file_path}")
        print(f"数据形状: {gdf.shape}")
        
        # 显示一些数据信息
        print("\n数据列信息:")
        print(gdf.columns.tolist())
        
        print("\n数据类型:")
        print(gdf.dtypes)
        
        print("\n空间参考系统:")
        print(gdf.crs)
        
        print("\n数据预览:")
        print(gdf.head())
        
        return gdf
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def visualize_geodata(gdf):
    """
    简单可视化地理数据
    """
    if gdf is not None:
        # 绘制地理数据
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax)
        plt.title('GeoPackage数据可视化')
        plt.savefig('geopackage_visualization.png')
        plt.show()
        print("已生成数据可视化")

# 使用示例
file_path = '/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result/1030008100.gpkg'
gdf = read_geopackage(file_path)

# 简单分析和统计
if gdf is not None:
    # 显示数据的基本统计信息
    print("\n数值列的基本统计信息:")
    numeric_columns = gdf.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        print(gdf[numeric_columns].describe())
    
    # 可视化数据
    visualize_geodata(gdf)
    
    # 获取GeoPackage中的所有图层
    import fiona
    with fiona.open(file_path) as src:
        print("\nGeoPackage中的图层:")
        print(src.schema)
        
        # 如果有多个图层,则列出所有图层
        try:
            layers = fiona.listlayers(file_path)
            print(f"图层列表: {layers}")
            
            # 遍历所有图层并打印基本信息
            for layer in layers:
                with fiona.open(file_path, layer=layer) as layer_src:
                    print(f"\n图层 '{layer}' 信息:")
                    print(f"记录数: {len(layer_src)}")
                    print(f"字段: {layer_src.schema['properties']}")
                    print(f"几何类型: {layer_src.schema['geometry']}")
        except Exception as e:
            print(f"获取图层信息时出错: {e}")