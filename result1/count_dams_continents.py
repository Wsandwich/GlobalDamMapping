import geopandas as gpd
import pandas as pd
import os
from tqdm import tqdm

# Define paths
continents_path = "E:/wyj/data/World_Continents_-530508453213467113.gpkg"
dams_dir = "E:/wyj/dam/v4/global_v5_shp"
years = [2010, 2015, 2020]

def main():
    print("Loading continent boundaries...")
    # Read continents from GeoPackage
    continents = gpd.read_file(continents_path, layer="World_Continents")
    print(f"Loaded {len(continents)} continents")
    print(f"Continent CRS: {continents.crs}")
    
    # Create a dictionary to store results
    all_continents = list(continents["CONTINENT"]) + ["Unknown"]
    results = pd.DataFrame(0, index=all_continents, columns=years)
    
    # Process each year
    for year in years:
        print(f"\nProcessing dams for year {year}...")
        dam_file = os.path.join(dams_dir, f"dams_{year}.shp")
        
        # Read the dam shapefile
        dams = gpd.read_file(dam_file)
        print(f"Loaded {len(dams)} dams for {year}")
        print(f"Dam CRS: {dams.crs}")
        
        # If dams don't have CRS, try to read from PRJ file or set it from the continent file
        if dams.crs is None:
            print("Dam shapefile has no CRS defined. Checking PRJ file...")
            prj_file = os.path.join(dams_dir, f"dams_{year}.prj")
            if os.path.exists(prj_file):
                # Read the PRJ file to get the CRS
                with open(prj_file, 'r') as f:
                    prj_text = f.read()
                print(f"Found PRJ file: {prj_text[:100]}...")
                try:
                    # Try to set the CRS from the PRJ file
                    dams = dams.set_crs(prj_text, allow_override=True)
                    print(f"Set CRS from PRJ file: {dams.crs}")
                except Exception as e:
                    print(f"Error setting CRS from PRJ file: {e}")
                    print("Setting CRS to WGS84 (EPSG:4326) as a fallback")
                    dams = dams.set_crs("EPSG:4326", allow_override=True)
            else:
                print("No PRJ file found, assuming WGS84 (EPSG:4326)")
                dams = dams.set_crs("EPSG:4326", allow_override=True)
        
        # Ensure both datasets have the same CRS
        if dams.crs != continents.crs:
            print(f"Reprojecting dams from {dams.crs} to match continent CRS ({continents.crs})")
            dams = dams.to_crs(continents.crs)
        
        # Create spatial index for continents to speed up the spatial join
        print("Creating spatial index for continents...")
        continents_sindex = continents.sindex
        
        # Use spatial join to count dams by continent
        print("Performing spatial join to count dams by continent...")
        joined = gpd.sjoin(dams, continents, how="left", predicate="within")
        counts = joined["CONTINENT"].value_counts().to_dict()
        
        # Add counts to results
        for continent, count in counts.items():
            if pd.notna(continent):
                results.loc[continent, year] = count
        
        # Count dams not in any continent (NaN values)
        nan_count = joined["CONTINENT"].isna().sum()
        results.loc["Unknown", year] = nan_count
        print(f"Dams outside any continent: {nan_count}")
    
    # Add total column and row
    results["Total"] = results.sum(axis=1)
    results.loc["Total"] = results.sum()
    
    # Print table
    print("\nNumber of dams in each continent by year:")
    print(results)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(dams_dir), "dam_counts_by_continent.csv")
    results.to_csv(output_path)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()