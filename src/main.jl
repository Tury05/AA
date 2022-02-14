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

