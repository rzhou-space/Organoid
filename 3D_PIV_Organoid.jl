using HDF5
using PyPlot
using multi_quickPIV
# using ImageView
using VTJK

# Function for loading and reading h5 files. 
function read_h5(folder_path::String, data_name::String)
    h5open(folder_path, "r") do file
        read(file, data_name)
    end
end

mask_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/organoid_10_mask.h5"
mask_vol_1 = Float32.(read_h5(mask_file_path, "tp42"))
PyPlot.imshow(mask_vol_1[:,:,32])

organoid_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/organoid_10_with_mask.h5"
vol_1 = Float32.(read_h5(organoid_file_path, "tp42")) # With size of (x, y, z)
vol_2 = Float32.(read_h5(organoid_file_path, "tp43"))



# # Min Max normalisation of 3D array. So that all images have the same range of intensities. 
# function min_max_normalize(arr::Array{T,3}) where T<:Real
#     min_val = minimum(arr)
#     max_val = maximum(arr)
#     normalized = (arr .- min_val) ./ (max_val - min_val)
#     return normalized
# end

# nor_vol1 = min_max_normalize(vol_1)
# nor_vol2 = min_max_normalize(vol_2)

PyPlot.imshow(vol_2[:, :, 32])
PyPlot.colorbar()

# # Reshape the two volumes so that they have the same shape for later processing with PIV. 
# # BUT: Not considering keeping the center the volume to be the same!
# function pad_to_match_shape(arr1::Array{T,3}, arr2::Array{T,3}) where T<:Real
#     # Determine target shape
#     x = max(size(arr1, 1), size(arr2, 1))
#     y = max(size(arr1, 2), size(arr2, 2))
#     z = max(size(arr1, 3), size(arr2, 3))
#     target_shape = (x, y, z)

#     # Helper function to pad a single array
#     function pad_array(arr::Array{T,3}, target_shape::Tuple{Int, Int, Int}) where T<:Real
#         padded = zeros(T, target_shape) # Pad the zeros to the enlarged space. 
#         padded[1:size(arr,1), 1:size(arr,2), 1:size(arr,3)] .= arr
#         return padded
#     end

#     return pad_array(arr1, target_shape), pad_array(arr2, target_shape)
# end

# # Padding two volumes. 
# pad_vol1, pad_vol2 = pad_to_match_shape(nor_vol1, nor_vol2) 
# # Since the original volume could be padded, the mask size has also to be fitted in. 
# pad_mask_1,      _ = pad_to_match_shape(mask_vol_1, pad_vol1)

pivparams = multi_quickPIV.setPIVParameters(interSize=(10, 10, 10), searchMargin=(20, 20, 20), step=(8, 8, 8), 
    mask_filtFun=(IA)->(IA[div.(size(IA),2)...]), mask_threshold=0.5)
# IA = div.(size(vol1), 5), ST = div.(IA,4), ST = max.(1, div.(IA,2))
# Apply the masked 3D PIV. 
VF, SN = multi_quickPIV.PIV(vol_1, vol_2, mask_vol_1, pivparams, precision=32)

IA = multi_quickPIV._isize(pivparams)
ST = multi_quickPIV._step(pivparams)

U = VF[1, :, :, :] # x component of vectors
V = VF[2, :, :, :] # y component of vectors 
W = VF[3, :, :, :] # z component of vectors 

ygrid = [(y-1)*ST[1] + div(IA[1], 2) for y in 1:size(U, 1), x in 1:size(U, 2), z in 1:size(U, 3)]
xgrid = [(x-1)*ST[2] + div(IA[2], 2) for y in 1:size(U, 1), x in 1:size(U, 2), z in 1:size(U, 3)]
zgrid = [(z-1)*ST[3] + div(IA[3], 2) for y in 1:size(U, 1), x in 1:size(U, 2), z in 1:size(U, 3)]

VTJK.vectorfield2VTK(U, V, W, fn="/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/test_piv", spacing=(8, 8, 8))

# Save the data into a .h5 file. 
organoid = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/organoid_10_with_mask_PIV"
# Save to .h5 with subgroups

h5writefile = string(organoid, ".h5") # The file name. 

h5open(h5writefile, "w") do file
    # Create top-level groups
    top_group = create_group(file, "tp42-43(step=8)")
    
    # Write datasets into groups
    write(top_group, "U", U)
    write(top_group, "V", V)
    write(top_group, "W", W)
    write(top_group, "xgrid", xgrid)
    write(top_group, "ygrid", ygrid)
    write(top_group, "zgrid", zgrid)
end



