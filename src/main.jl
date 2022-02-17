using DelimitedFiles;
using Statistics;

maxMinNorm = function (v, min, max)
	return (v .- min)./(max .- min);
end;

media0Norm = function (v, avg, std)
	return (v .- avg)./std;
end;

dataset = readdlm("./BBDD/iris/iris.data", ',');
f, c = size(dataset);

inDS = dataset[:, 1:c-1];
inDS[:, 1] .= 1; 
outDS = dataset[:, c];
categOutDS = unique(outDS);

@assert length(categOutDS) > 2

target = if length(categOutDS) == 2
		categOutDS[1] .== outDS
	else
		map(x -> x .== categOutDS, outDS)
	end;

maxIn = maximum(inDS, dims=1);
minIn = minimum(inDS, dims=1);
avgIn = mean(inDS, dims=1);
stdIn = std(inDS, dims=1);

indexNullColumn = last.(Tuple.(findall((maxIn .== minIn) .* (stdIn .== 0))));

if !isempty(indexNullColumn)
	c-=1;
	inDS = inDS[:, 1:end .!= indexNullColumn];
	maxIn = maxIn[:, 1:end .!= indexNullColumn];
	minIn = minIn[:, 1:end .!= indexNullColumn];
	avgIn = avgIn[:, 1:end .!= indexNullColumn];
	stdIn = stdIn[:, 1:end .!= indexNullColumn];
end;

normWithMaxMin = 0.75 .> maxMinNorm(avgIn, minIn, maxIn) .> 0.25;

inputs = Array{Float32, 2}(undef, f, c-1);

for i in 1:(c-1)
	inputs[:, i] = if(normWithMaxMin[i])
		maxMinNorm(inDS[:, i], minIn[i], maxIn[i])
	else
		media0Norm(inDS[:, i], avgIn[i], stdIn[i])
	end;
end;

#1
oneHotEncoding = function (feature::AbstractArray{<:Any,1},
		classes::AbstractArray{<:Any,1}=[])
end;

oneHotEncoding = function (feature::AbstractArray{<:Bool,1})
end;


#2
calculateMinMaxNormalizationParameters = function (inputs::AbstractArray{<:Real,2})
	(minimum(inputs, dims=1), maximum(inputs, dims=1))
end

calculateZeroMeanNormalizationParameters = function (inputs::AbstractArray{<:Real,2})
	(mean(inputs, dims=1), std(inputs, dims=1))
end


normalizeMinMax! = function (inputs::AbstractArray{Float32,2},
	minMax::NTuple{2, AbstractArray{<:Real,2}})
	_, x = size(inputs)
	for i in 1:x
		inputs[:, i] = maxMinNorm(inputs[:, i], minMax[1][i], minMax[2][i])
	end
end

normalizeMinMax! = function (inputs::AbstractArray{Float32,2})
	_, x = size(inputs)
	minMax = calculateMinMaxNormalizationParameters(inputs)
	for i in 1:x
		inputs[:, i] = maxMinNorm(inputs[:, i], minMax[1][i], minMax[2][i])
	end
end

normalizeMinMax = function (inputs::AbstractArray{Float32,2},
	minMax::NTuple{2, AbstractArray{<:Real,2}}=())
	out = copy(inputs)
	_, x = size(out)
	for i in 1:x
		out[:, i] = maxMinNorm(out[:, i], minMax[1][i], minMax[2][i])
	end
	return out
end

normalizeMinMax = function (inputs::AbstractArray{Float32,2})
	out = copy(inputs)
	minMax = calculateMinMaxNormalizationParameters(out)
	_, x = size(out)
	for i in 1:x
		out[:, i] = maxMinNorm(out[:, i], minMax[1][i], minMax[2][i])
	end
	return out
end


normalizeZeroMean! = function (inputs::AbstractArray{Float32,2},
		meanStd::NTuple{2, AbstractArray{<:Real,2}})
	_, x = size(inputs)
	for i in 1:x
		inputs[:, i] = media0Norm(inputs[:, i], meanStd[1][i], meanStd[2][i])
	end
end

normalizeZeroMean! = function (inputs::AbstractArray{Float32,2})
	meanStd = calculateZeroMeanNormalizationParameters(inputs)
	_, x = size(inputs)
	for i in 1:x
		inputs[:, i] = media0Norm(inputs[:, i], meanStd[1][i], meanStd[2][i])
	end
end

normalizeZeroMean = function (inputs::AbstractArray{Float32,2},
		meanStd::NTuple{2, AbstractArray{<:Real,2}})
	out = copy(inputs)
	_, x = size(out)
	for i in 1:x
		out[:, i] = media0Norm(out[:, i], meanStd[1][i], meanStd[2][i])
	end
	return out
end

normalizeZeroMean = function (inputs::AbstractArray{Float32,2})
	out = copy(inputs)
	meanStd = calculateZeroMeanNormalizationParameters(out)
	_, x = size(out)
	for i in 1:x
		out[:, i] = media0Norm(out[:, i], meanStd[1][i], meanStd[2][i])
	end
	return out
end

#3 (dificultad media)
classifyOutputs = function (outputs::AbstractArray{<:Real,2})
	
end;


#4 (dificultad media)
accuracy = function (target::AbstractArray{Bool,1},
		outputs::AbstractArray{Bool,1})
	
end;

accuracy = function (target::AbstractArray{Bool,2},
		outputs::AbstractArray{Bool,2})
	
end;

accuracy = function (target::AbstractArray{Bool,1},
		outputs::AbstractArray{<:Real,1})
	
end;

accuracy = function (target::AbstractArray{Bool,2},
		outputs::AbstractArray{<:Real,2})
	
end;


#5 (dificultad alta)
classRNA = function (topology::AbstractArray{<:Int,1},
		funActivacion::AbstractArray{Function,1},
		dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
		maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
	
end;

entrenarClassRNA = function (topology::AbstractArray{<:Int,1},
		dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
		maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
	
end;

entrenarClassRNA = function (topology::AbstractArray{<:Int,1},
		dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
		maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
	
end;


#6 (probar las funciones)
