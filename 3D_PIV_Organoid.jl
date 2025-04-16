using HDF5
using PyPlot
using multi_quickPIV

# Function for loading and reading h5 files. 
function read_h5(folder_path::String, data_name::String)
    h5open(folder_path, "r") do file
        read(file, data_name)
    end
end

file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/predict_h5/organoid_10.h5"
vol_1 = Float16.(read_h5(file_path, "tp42"))
vol_2 = Float16.(read_h5(file_path, "tp43"))

# Min Max normalisation of 3D array.
function min_max_normalize(arr::Array{T,3}) where T<:Real
    min_val = minimum(arr)
    max_val = maximum(arr)
    normalized = (arr .- min_val) ./ (max_val - min_val)
    return normalized
end

nor_vol1 = min_max_normalize(vol_1)
nor_vol2 = min_max_normalize(vol_2)

PyPlot.imshow(nor_vol1[:, :, 10])
PyPlot.colorbar()

# Reshape the two volumes so that they have the same shape for later processing with PIV. 

function pad_to_match_shape(arr1::Array{T,3}, arr2::Array{T,3}) where T<:Real
    # Determine target shape
    x = max(size(arr1, 1), size(arr2, 1))
    y = max(size(arr1, 2), size(arr2, 2))
    z = max(size(arr1, 3), size(arr2, 3))
    target_shape = (x, y, z)

    # Helper function to pad a single array
    function pad_array(arr::Array{T,3}, target_shape::Tuple{Int, Int, Int}) where T<:Real
        padded = zeros(T, target_shape)
        padded[1:size(arr,1), 1:size(arr,2), 1:size(arr,3)] .= arr
        return padded
    end

    return pad_array(arr1, target_shape), pad_array(arr2, target_shape)
end

pad_vol1, pad_vol2 = pad_to_match_shape(nor_vol1, nor_vol2)

PyPlot.imshow(pad_vol1[:, :, 10])

pivparams = multi_quickPIV.setPIVParameters(interSize=(8, 8, 8), searchMargin=(16, 16, 16), step=(8, 8, 8))
VF, SN = multi_quickPIV.PIV(pad_vol1, pad_vol2, pivparams, precision=16)

IA = multi_quickPIV._isize(pivparams)
ST = multi_quickPIV._step(pivparams)

U = VF[1, :, :, :] # x component of vectors
V = VF[2, :, :, :] # y component of vectors 
W = VF[3, :, :, :] # z component of vectors 

# TODO: Visualisation the vector field in the Paraview overlap with organoid. 
# TODO: Add mask on the volume and then do PIV. 
