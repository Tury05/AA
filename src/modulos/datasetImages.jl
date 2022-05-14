using DelimitedFiles;
using JLD2;
using Images;
using Statistics;
using Random;
using Plots;
using Flux;
using Flux.Losses;

# Functions that allow the conversion from images to Float64 arrays
imageToGrayArray(image:: Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float64,2}, gray.(Gray.(image)));
imageToGrayArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToGrayArray(RGB.(image));
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float32, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

# Function to read all of the images in a folder and return them as 2 Float64 arrays: one with color components (3D array) and the other with grayscale components (2D array)
function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            # Check that they are color images
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            # Add the image to the vector of images
            push!(images, image);
        end;
    end;
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return (imageToColorArray.(images), imageToGrayArray.(images));
end;

# Functions to load the dataset
function loadTrainingDataset(positiveFolderName::String, negativeFolderName::String)
    (positivesColor, positivesGray) = loadFolderImages(positiveFolderName);
    (negativesColor, negativesGray) = loadFolderImages(negativeFolderName);
    targets = [trues(length(positivesColor)); falses(length(negativesColor))];
    return ([positivesColor; negativesColor], [positivesGray; negativesGray], targets);
end;

loadTestDataset() = ((colorMatrix,_) = loadFolderImages("test"); return colorMatrix; );

function loadImage(fileName::String)
	if  any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"])
        image = load(fileName);
        # Check that they are color images
        @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))

        return (imageToColorArray(image), imageToGrayArray(image));
    else
    	throw("No es una imagen");
    end;
end;

function imageToData(fileName::String)

	image,_ = loadImage(fileName);

	data = Array{Float64, 1}(undef, 9);

	fil = convert(Int,round(size(image,1)/7));
	col = convert(Int,round(size(image,2)/3));

	k = 1
	for j in 1:3
		for c in 1:3
			data[k] = mean(image[(fil*(j*2-1)):(fil*(j*2)),col:(col*2),c]);
			k += 1;
		end;
	end;

	return data;
end;

function imageToData2(filename::String)
	
	data = imageToData(filename);
	
	data2 = Array{Float64, 1}(undef, 7);
	
	conj = (data[1], data[2], data[3]);
	conj2 = (data[7], data[8], data[9]);
	
	data2[1] = mean(conj);
	data2[2] = std(conj);
	data2[3] = data[4];
	data2[4] = data[5];
	data2[5] = data[6];
	data2[6] = mean(conj2);
	data2[7] = std(conj2);
	
	return data2;
end;
	
function imageToData3(filename::String)
	data = imageToData(filename)
	
	data3 = Array{Float64, 1}(undef, 5);
	
	conj = (data[1], data[2], data[3], data[7], data[8], data[9]);
	
	data3[1] = mean(conj);
	data3[2] = std(conj);
	data3[3] = data[4];
	data3[4] = data[5];
	data3[5] = data[6];
	
	return data3;
end;

function santaImagesToDatasets(santaFolder::String, notSantaFolder::String)

	santaImages,_ = loadFolderImages(santaFolder);
	notSantaImages,_ = loadFolderImages(notSantaFolder);
	images = (santaImages, notSantaImages);

	santaDataset = Array{Float64, 2}(undef, size(santaImages,1), 9);
	notSantaDataset = Array{Float64, 2}(undef, size(notSantaImages,1), 9);
	datasets = (santaDataset, notSantaDataset);

	for s in 1:2
		i = 1
		for colorMatrix in images[s]

			fil = convert(Int,round(size(colorMatrix,1)/7));
			col = convert(Int,round(size(colorMatrix,2)/3));

			k = 1
			for j in 1:3
				for c in 1:3
					datasets[s][i,k] = mean(colorMatrix[(fil*(j*2-1)):(fil*(j*2)),col:(col*2),c]);
					k += 1;
				end;
			end;

			i += 1;
		end;
	end;

	return datasets;
end;



function eyeImagesToDatasets(colorMatrix::AbstractArray{Float64,2})

	fil = convert(Int,floor(size(colorMatrix,1)/3));
	col = convert(Int,floor(size(colorMatrix,2)/7));
	dataset = Array{Float64, 1}(undef, 3);

	for j in 1:3
		
		dataset[j] = mean(colorMatrix[fil:(fil*2),(col*(j*2-1)):(col*(j*2))]);
	end;

	return dataset;
end;

function eyeImagesToDatasets(eyeFolder::String, notEyeFolder::String)

	_, eyeImages = loadFolderImages(eyeFolder);
	_, notEyeImages = loadFolderImages(notEyeFolder);
	images = (eyeImages, notEyeImages);

	eyeDataset = Array{Float64, 2}(undef, size(eyeImages,1), 3);
	notEyeDataset = Array{Float64, 2}(undef, size(notEyeImages,1), 3);
	datasets = (eyeDataset, notEyeDataset);

	for s in 1:2
		i = 1
		for colorMatrix in images[s]
			datasets[s][i,:] = eyeImagesToDatasets(colorMatrix);

			i += 1;
		end;
	end;

	return datasets;
end;

function testRNAfaceImage(faceImage::String, rna::Chain{Tuple{Dense{typeof(σ),Array{Float32,2},Array{Float32,1}},Dense{typeof(σ),Array{Float32,2},Array{Float32,1}},Dense{typeof(σ),Array{Float32,2},Array{Float32,1}}}},
	uRNA::Real, uDist::Real, uErrors::Int)

	_,image = loadImage(faceImage);

	fil = convert(Int,round(size(image,1)/60));
	col = convert(Int,round(size(image,2)/64));

	ojoIXs = [];
	ojoIYs = [];
	maxDist = 19.42*uDist;
	nErrors = 0;

	for y in 20:36
		for x in 16:27
			patron = eyeImagesToDatasets(image[(y*fil):((y+3)*fil),(x*col):((x+4)*col)]);
			if (rna(patron)[1] > uRNA) & (rna(reverse(patron, dims=1))[1] > uRNA)
				if !isempty(ojoIYs)
					if sqrt((mean(ojoIYs)-y)^2 + (mean(ojoIXs)-x)^2) < maxDist
						push!(ojoIYs, y);
						push!(ojoIXs, x);
					else
						if nErrors > uErrors
							return false;
						else
							nErrors+=1;
						end;
					end;
				else
					push!(ojoIYs, y);
					push!(ojoIXs, x);
				end;
			end;
		end;
	end;

	ojoDXs = [];
	ojoDYs = [];
	nErrors = 0;

	for y in 20:36
		for x in 16:27
			patron = eyeImagesToDatasets(image[(y*fil):((y+3)*fil),(x*col):((x+4)*col)]);
			if (rna(patron)[1] > uRNA) & (rna(reverse(patron, dims=1))[1] > uRNA)
				if !isempty(ojoDYs)
					if sqrt((mean(ojoDYs)-y)^2 + (mean(ojoDXs)-x)^2) < maxDist
						push!(ojoDYs, y);
						push!(ojoDXs, x);
					else
						if nErrors > uErrors
							return false;
						else
							nErrors+=1;
						end;
					end;
				else
					push!(ojoDYs, y);
					push!(ojoDXs, x);
				end;
			end;
		end;
	end;

	return !isempty(ojoIYs) & !isempty(ojoDYs);
end;

function santaImagesToDatasets2(santaFolder::String, notSantaFolder::String)

	datasets = santaImagesToDatasets(santaFolder, notSantaFolder);
	
	santaDataset2 = Array{Float64, 2}(undef, size(datasets[1],1), 7);
	notsantaDataset2 = Array{Float64, 2}(undef, size(datasets[2],1), 7);
	datasets2 = (santaDataset2, notsantaDataset2);
	for s in 1:2
		i = 1
		for img in datasets[s]
			conj1 = (datasets[s][i,1], datasets[s][i,2], datasets[s][i,3]);
			conj2 = (datasets[s][i,7], datasets[s][i,8], datasets[s][i,9]);
			datasets2[s][i, 1] = mean(conj1);
			datasets2[s][i, 2] = std(conj2);
			datasets2[s][i, 3] = datasets[s][i,4];
			datasets2[s][i, 4] = datasets[s][i,5];
			datasets2[s][i, 5] = datasets[s][i,6];
			datasets2[s][i, 6] = mean(conj2);
			datasets2[s][i, 7] = std(conj2);
			
			i=+ 1;
		end;
	end;
	
	return datasets2;
end;

function santaImagesToDatasets3(santaFolder::String, notSantaFolder::String)

	datasets = santaImagesToDatasets(santaFolder, notSantaFolder);
	
	santaDataset3 = Array{Float64, 2}(undef, size(datasets[1], 1), 5);
	notsantaDataset3 = Array{Float64, 2}(undef, size(datasets[2], 1), 5);
	datasets3 = (santaDataset3, notsantaDataset3);
	
	for s in 1:2
		
		for i in 1:size(datasets[s],1)
			conj = (datasets[s][i,1], datasets[s][i,2],  datasets[s][i,3], datasets[s][i,7], datasets[s][i,8], datasets[s][i,9]);
			
			datasets3[s][i,1] = mean(conj);
			datasets3[s][i,2] = std(conj);
			datasets3[s][i, 3] = datasets[s][i,4];
			datasets3[s][i, 4] = datasets[s][i,5];
			datasets3[s][i, 5] = datasets[s][i,6];
			
			
		end;
	end;
	
	return datasets3;
end;

function randDataset(a1::AbstractArray{Float64,2}, a2::AbstractArray{Float64,2})
	
	@assert(size(a1) == size(a2)); 
	inDSLength = size(a1,1)*2;
	perm = randperm(inDSLength);

	inDS = Array{Float64, 2}(undef, inDSLength, size(a1,2));

	outDS = Array{Bool}(undef, inDSLength);
	k = 1;

	for i in 1:size(a1,1)
		inDS[perm[k],:] = a1[i,:];
		outDS[perm[k]] = true;
		k += 1;
	end;

	for i in 1:size(a2,1)
		inDS[perm[k],:] = a2[i,:];
		outDS[perm[k]] = false;
		k += 1;
	end;

	return (inDS, outDS);
end;
