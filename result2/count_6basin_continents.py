import geopandas as gpd
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def main():
    # Define paths
    basins_path = r"E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev06_with_dam_stats_continent.shp"
    output_dir = r"E:\wyj\project\dam\result1\output\continent_stats"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading basins with continent information...")
    # Read basins shapefile (assuming it already has the CONTINENT attribute from previous step)
    basins = gpd.read_file(basins_path)
    print(f"Loaded {len(basins)} basins")
    
    # Define dam type mapping
    dam_type_mapping = {
        'e': 'embankment',  # 土石坝
        'g': 'gravity',     # 重力坝
        'b': 'Barrage',     # 闸坝
        'a': 'arch'         # 拱坝
    }
    
    # Calculate changes for each dam type (2020 - 2010) if not already calculated
    for code, full_name in dam_type_mapping.items():
        change_col = f'change_{code}'
        if change_col not in basins.columns:
            basins[change_col] = basins[f'2020_{code}'] - basins[f'2010_{code}']
            print(f"Calculated change for {full_name} dams: {change_col}")
    
    # Calculate total change if not already calculated
    change_columns = [f'change_{code}' for code in dam_type_mapping.keys()]
    if 'change_total' not in basins.columns:
        basins['change_total'] = basins[change_columns].sum(axis=1)
        print("Calculated total change")
    
    # Create dictionary to store results by continent
    continent_stats = {}
    
    # Get list of all continents
    continents = basins['CONTINENT'].dropna().unique().tolist()
    print(f"Found {len(continents)} continents: {continents}")
    
    # Create DataFrames to store results
    increase_df = pd.DataFrame(index=continents, columns=list(dam_type_mapping.values()) + ['Total'])
    decrease_df = pd.DataFrame(index=continents, columns=list(dam_type_mapping.values()) + ['Total'])
    no_change_df = pd.DataFrame(index=continents, columns=list(dam_type_mapping.values()) + ['Total'])
    
    # Create all_stats to hold complete statistics
    all_stats = pd.DataFrame(
        index=continents + ['Global'], 
        columns=pd.MultiIndex.from_product([
            list(dam_type_mapping.values()) + ['Total'],
            ['Increase', 'Decrease', 'No Change', 'Net Change']
        ])
    )
    
    # Count basins with increasing, decreasing, and no change for each dam type by continent
    print("Analyzing dam changes by continent...")
    for continent in continents:
        continent_basins = basins[basins['CONTINENT'] == continent]
        
        # Process each dam type
        for code, full_name in dam_type_mapping.items():
            change_col = f'change_{code}'
            
            # Count increases, decreases, and no changes
            increases = (continent_basins[change_col] > 0).sum()
            decreases = (continent_basins[change_col] < 0).sum()
            no_changes = (continent_basins[change_col] == 0).sum()
            
            # Store in DataFrames
            increase_df.at[continent, full_name] = increases
            decrease_df.at[continent, full_name] = decreases
            no_change_df.at[continent, full_name] = no_changes
            
            # Store in all_stats
            all_stats.at[continent, (full_name, 'Increase')] = increases
            all_stats.at[continent, (full_name, 'Decrease')] = decreases
            all_stats.at[continent, (full_name, 'No Change')] = no_changes
            all_stats.at[continent, (full_name, 'Net Change')] = increases - decreases
        
        # Process total change
        increases = (continent_basins['change_total'] > 0).sum()
        decreases = (continent_basins['change_total'] < 0).sum()
        no_changes = (continent_basins['change_total'] == 0).sum()
        
        # Store in DataFrames
        increase_df.at[continent, 'Total'] = increases
        decrease_df.at[continent, 'Total'] = decreases
        no_change_df.at[continent, 'Total'] = no_changes
        
        # Store in all_stats
        all_stats.at[continent, ('Total', 'Increase')] = increases
        all_stats.at[continent, ('Total', 'Decrease')] = decreases
        all_stats.at[continent, ('Total', 'No Change')] = no_changes
        all_stats.at[continent, ('Total', 'Net Change')] = increases - decreases
    
    # Calculate global statistics
    for code, full_name in dam_type_mapping.items():
        change_col = f'change_{code}'
        
        # Count global increases, decreases, and no changes
        global_increases = (basins[change_col] > 0).sum()
        global_decreases = (basins[change_col] < 0).sum()
        global_no_changes = (basins[change_col] == 0).sum()
        
        # Store in all_stats
        all_stats.at['Global', (full_name, 'Increase')] = global_increases
        all_stats.at['Global', (full_name, 'Decrease')] = global_decreases
        all_stats.at['Global', (full_name, 'No Change')] = global_no_changes
        all_stats.at['Global', (full_name, 'Net Change')] = global_increases - global_decreases
    
    # Process global total change
    global_increases = (basins['change_total'] > 0).sum()
    global_decreases = (basins['change_total'] < 0).sum()
    global_no_changes = (basins['change_total'] == 0).sum()
    
    # Store in all_stats
    all_stats.at['Global', ('Total', 'Increase')] = global_increases
    all_stats.at['Global', ('Total', 'Decrease')] = global_decreases
    all_stats.at['Global', ('Total', 'No Change')] = global_no_changes
    all_stats.at['Global', ('Total', 'Net Change')] = global_increases - global_decreases
    
    # Save all results to CSV files
    print("Saving results to CSV files...")
    increase_df.to_csv(os.path.join(output_dir, 'dam_increases_by_continent.csv'))
    decrease_df.to_csv(os.path.join(output_dir, 'dam_decreases_by_continent.csv'))
    no_change_df.to_csv(os.path.join(output_dir, 'dam_no_changes_by_continent.csv'))
    all_stats.to_csv(os.path.join(output_dir, 'dam_all_stats_by_continent.csv'))
    
    # Add percentage calculations to all_stats
    all_stats_pct = all_stats.copy()
    
    # Create a new DataFrame for percentages
    for continent in continents + ['Global']:
        for dam_type in list(dam_type_mapping.values()) + ['Total']:
            total_basins = all_stats.loc[continent, (dam_type, 'Increase')] + \
                          all_stats.loc[continent, (dam_type, 'Decrease')] + \
                          all_stats.loc[continent, (dam_type, 'No Change')]
            
            if total_basins > 0:
                # Calculate percentages
                all_stats_pct.loc[continent, (dam_type, 'Increase')] = \
                    (all_stats.loc[continent, (dam_type, 'Increase')] / total_basins) * 100
                
                all_stats_pct.loc[continent, (dam_type, 'Decrease')] = \
                    (all_stats.loc[continent, (dam_type, 'Decrease')] / total_basins) * 100
                
                all_stats_pct.loc[continent, (dam_type, 'No Change')] = \
                    (all_stats.loc[continent, (dam_type, 'No Change')] / total_basins) * 100
    
    # Save percentage results
    all_stats_pct.to_csv(os.path.join(output_dir, 'dam_all_stats_percentages_by_continent.csv'))
    
    # Print summary
    print("\nSummary of Dam Changes by Continent:")
    print("\nIncreases:")
    print(increase_df)
    print("\nDecreases:")
    print(decrease_df)
    print("\nNet Change (Increases - Decreases):")
    net_change = increase_df - decrease_df
    print(net_change)
    
    print("\nDone! Results saved to:", output_dir)

if __name__ == "__main__":
    main()