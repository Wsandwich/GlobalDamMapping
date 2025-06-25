import geopandas as gpd

# 加载Shapefile
rhine_dis = gpd.read_file(r"C:\Users\86157\Documents\RHINE_DIS_2010.shp")
rhine_dam = gpd.read_file(r"C:\Users\86157\Documents\RHINE_DAM_2010.shp")

# 确保FID和layer列为相同类型（如果不是，可以转换类型）
rhine_dis['FID'] = rhine_dis['FID'].astype(int)
rhine_dam['FID'] = rhine_dam['FID'].astype(int)

# 如果layer列存在且数据类型需要标准化
rhine_dis['layer'] = rhine_dis['layer'].astype(str)
rhine_dam['layer'] = rhine_dam['layer'].astype(str)

# 使用FID和layer作为合并的键
rhine_merged = rhine_dam.merge(rhine_dis[['FID', 'layer', 'HubName']], on=['FID', 'layer'], how='left')

# 重命名HubName为GOID
rhine_merged.rename(columns={'HubName': 'GOID'}, inplace=True)

# 保存新的Shapefile
rhine_merged.to_file(r"C:\Users\86157\Documents\RHINE_MERGED_2010.shp")
