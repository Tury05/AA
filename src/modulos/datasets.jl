using DelimitedFiles;

maxMinNorm = function (v, min, max)
	return (v .- min)./(max .- min);
end;

media0Norm = function (v, avg, std)
	return (v .- avg)./std;
end;

function readData(dataset)
	data = readdlm(dataset, ',')
	f, c = size(data);

	inDS = data[:, 1:c-1];
	outDS = data[:, c];
	return inDS, outDS
end

function oneHotEncoding(feature::AbstractArray{<:Any,1},classes::AbstractArray{<:Any,1}=[])
	if length(classes) == 2
		out = classes[1] .== feature
	else
		out = Array{Bool, 2}(undef, length(feature), length(classes))
		for i in 1:size(out, 1)
			out[i, :] = feature[i].==classes
		end
	end
	return out
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

function oneHotEncoding(feature::AbstractArray{<:Bool,1})
	m = reshape(feature, :, 1)
	return m
end;

calculateMinMaxNormalizationParameters = function (inputs::AbstractArray{<:Real,2})
	(minimum(inputs, dims=1), maximum(inputs, dims=1))
end

calculateZeroMeanNormalizationParameters = function (inputs::AbstractArray{<:Real,2})
	(mean(inputs, dims=1), std(inputs, dims=1))
end


function normalizeMinMax!(inputs::AbstractArray{Float32,2},
	minMax::NTuple{2, AbstractArray{<:Real,2}})
	for i in 1:size(inputs,2)
		inputs[:, i] = maxMinNorm(inputs[:, i], minMax[1][i], minMax[2][i])
	end
end

function normalizeMinMax!(inputs::AbstractArray{Float32,2})
	minMax = calculateMinMaxNormalizationParameters(inputs)
	for i in 1:size(inputs,2)
		inputs[:, i] = maxMinNorm(inputs[:, i], minMax[1][i], minMax[2][i])
	end
end

function normalizeMinMax(inputs::AbstractArray{Float32,2},
	minMax::NTuple{2, AbstractArray{<:Real,2}}=())
	out = copy(inputs)
	for i in 1:size(out,2)
		out[:, i] = maxMinNorm(out[:, i], minMax[1][i], minMax[2][i])
	end
	return out
end

function normalizeMinMax(inputs::AbstractArray{Float32,2})
	out = copy(inputs)
	minMax = calculateMinMaxNormalizationParameters(out)
	for i in 1:size(out,2)
		out[:, i] = maxMinNorm(out[:, i], minMax[1][i], minMax[2][i])
	end
	return out
end


function normalizeZeroMean!(inputs::AbstractArray{Float32,2},
		meanStd::NTuple{2, AbstractArray{<:Real,2}})
	for i in 1:size(inputs,2)
		inputs[:, i] = media0Norm(inputs[:, i], meanStd[1][i], meanStd[2][i])
	end
end

function normalizeZeroMean!(inputs::AbstractArray{Float32,2})
	meanStd = calculateZeroMeanNormalizationParameters(inputs)
	for i in 1:size(inputs,2)
		inputs[:, i] = media0Norm(inputs[:, i], meanStd[1][i], meanStd[2][i])
	end
end

function normalizeZeroMean(inputs::AbstractArray{Float32,2},
		meanStd::NTuple{2, AbstractArray{<:Real,2}})
	out = copy(inputs)
	for i in 1:size(out,2)
		out[:, i] = media0Norm(out[:, i], meanStd[1][i], meanStd[2][i])
	end
	return out
end

function normalizeZeroMean(inputs::AbstractArray{Float32,2})
	out = copy(inputs)
	meanStd = calculateZeroMeanNormalizationParameters(out)
	for i in 1:size(out,2)
		out[:, i] = media0Norm(out[:, i], meanStd[1][i], meanStd[2][i])
	end
	return out
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}, threshold = 0.5)
	if size(outputs, 2) == 1
		out = outputs .>= threshold
	else
		out = falses(size(outputs))
		(_,indicesMaxEachInstance) = findmax(outputs, dims=2)
		out[indicesMaxEachInstance] .= true
	end
	return out
end
