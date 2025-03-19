#!/bin/bash
# Regrids from tripolar grid to PlateCarree (1x1 degree resolution)
# Both bilinear and conservative remapping is done and the regridded files are saved in two different files 

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <grid_file.nc> <data_file.nc>"
    exit 1
fi

# Assign input arguments to variables
grid_file=$1
data_file=$2

echo "Regridding $data_file using grid from $grid_file..."

# Step 1: Add plat and plon from the grid file to the data file
ncks -A -v plat,plon "$grid_file" "$data_file"

# Step 2: Rearrange dimensions and extract pclat and pclon
ncpdq -a y,x,nv -v pclat,pclon "$grid_file" grid_tmp.nc

# Step 3: Add pclat and pclon to the data file
ncks -A -v pclat,pclon grid_tmp.nc "$data_file"

# Step 4: Add bounds attributes to plon and plat
ncatted -a bounds,plon,c,c,'pclon' -a bounds,plat,c,c,'pclat' "$data_file"

# Step 5: Perform bilinear regridding
output_file_bil="${data_file%.nc}_bil.nc"
cdo remapbil,global_1 "$data_file" "$output_file_bil"

echo "Bilinear regridding completed: $output_file_bil"

# Step 6: Perform conservative regridding
output_file_con="${data_file%.nc}_con.nc"
cdo remapcon,global_1 "$data_file" "$output_file_con"

# Cleanup temporary files
rm -f grid_tmp.nc

echo "Conservative regridding completed: $output_file_con"
